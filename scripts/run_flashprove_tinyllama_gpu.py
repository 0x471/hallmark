"""Commit to and prove dense TinyLlama on one rented sm_90 GPU under a spend cap; recheck here.

usage: python scripts/run_flashprove_tinyllama_gpu.py --credit-balance-usd USD --max-cost-usd USD
       --build DIR --gguf FILE [--proofs N] [--gpu NAME] [--cloud SECURE|COMMUNITY]

One paid host, one attempt, no retry. Before anything is paid for, the caller's enrollment is
written (enrollment.json: the kit, the request, the commitment ID and the patched verifier's
digest that this caller will accept), the pinned tokenizer and chat template must give the
request's ids for the case's text, the patched and the unpatched cp_verify are built here from
the pinned source, and the worst case of the charge (the hourly price times every time limit) must
fit both the cap and the balance that the caller states. The host runs gpu_job.sh on public files
only; the provider's secret directory never leaves it and the host is deleted in every case: by
this process, by a second process that outlives it, and by the provider at the time limit. Then
scripts/flashprove_tinyllama_proof.py's checks run here on the returned files, under the enrolled
commitment ID: a commitment with another ID fails the run.

DIR holds the patched programs hc/cp_commit, hc/cp_prove and hc/cp_verify (upstream's
prover/build.sh and verifier/build.sh on the pinned source with the four patches applied; this
script does not build them). FILE is the exported model, the one that kit.json pins (the answer
target rebuilds it). The provider's API key is read from the environment (RUNPOD_API_KEY) and is
never written anywhere.

The cap is the caller's own: this script makes no claim that an owner approved it. Its spend
figure is a bound from the provider's hourly price and this process's clock, with the fall of the
account's balance beside it; neither is an invoice.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from decimal import Decimal, InvalidOperation
from pathlib import Path

import flashprove as fp
import flashprove_tinyllama as ft
import flashprove_tinyllama_proof as proof
import flashprove_tinyllama_request as request_text
from run_e2e import sha, write_json
from run_llm_e2e import require

ROOT, PROBE = ft.ROOT, ft.PROBE
RUNS = ROOT / "runs"
ENROLLMENT = proof.ENROLLED
POD = json.loads((PROBE / "pod.json").read_text())
LIMITS = POD["seconds"]  # one time limit for each paid step
BASE = "/var/tmp/fp"
PROGRAMS = ("cp_commit", "cp_prove", "cp_verify")
RETURNED = (  # what comes back besides public/: logs and records, nothing of isk/, home/ or tmp/
    "commitment-id.txt",
    "cp_commit.log",
    "cp_commit.time",
    "cp_commit.status",
    "worst-case-table.txt",
    "binaries.sha256",
    "inputs.sha256",
    "public.sha256",
    "gpu.txt",
    "cpu.txt",
    "memory.txt",
)


class RunPod:
    """The four calls this controller makes of the provider. The key stays in this object."""

    URL = "https://api.runpod.io/graphql"

    def __init__(self, key):
        require(bool(key), "RUNPOD_API_KEY is not set")
        self._key = key

    def _call(self, query, variables=None):
        request = urllib.request.Request(
            self.URL,
            data=json.dumps(dict(query=query, variables=variables or {})).encode(),
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
                "User-Agent": "hallmark-flashprove/1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as reply:
                value = json.loads(reply.read().decode())
        except urllib.error.HTTPError as error:
            raise ValueError(f"the provider answered {error.code}") from None
        if "errors" in value or "data" not in value:
            said = str((value.get("errors") or [{}])[0].get("message", ""))[:300]
            raise ValueError("the provider refused the call: " + said)
        return value["data"]

    def account(self):
        """(balance in USD, the ids of the pods that exist)."""
        me = self._call("query { myself { clientBalance pods { id } } }")["myself"]
        return Decimal(str(me["clientBalance"])), [pod["id"] for pod in me["pods"]]

    def price(self, gpu, cloud):
        """USD an hour for one such GPU, by the provider's own list."""
        types = self._call("query { gpuTypes { id securePrice communityPrice } }")["gpuTypes"]
        (row,) = [t for t in types if t["id"] == gpu] or [None]
        require(row is not None, "the provider lists no such GPU")
        value = row["securePrice" if cloud == "SECURE" else "communityPrice"]
        require(value is not None and value > 0, "the provider gives no price for this GPU")
        return Decimal(str(value))

    def create(self, spec):
        fields = "id costPerHr gpuCount vcpuCount memoryInGb machine { gpuDisplayName }"
        query = (
            "mutation($input: PodFindAndDeployOnDemandInput) "
            "{ podFindAndDeployOnDemand(input: $input) { " + fields + " } }"
        )
        pod = self._call(query, {"input": spec})["podFindAndDeployOnDemand"]
        require(pod is not None and pod.get("id"), "the provider made no host")
        return pod

    def address(self, pod_id):
        """(public address, port) of the host's SSH service, or None while it starts."""
        query = (
            "query { myself { pods { id runtime "
            "{ ports { ip isIpPublic privatePort publicPort } } } } }"
        )
        for pod in self._call(query)["myself"]["pods"]:
            for port in (pod.get("runtime") or {}).get("ports") or []:
                if pod["id"] == pod_id and port["privatePort"] == 22 and port["isIpPublic"]:
                    return port["ip"], int(port["publicPort"])
        return None

    def terminate(self, pod_id):
        self._call("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})


class Remote:
    """SSH and scp to the rented host with a key made for this run alone."""

    def __init__(self, output, host, port):
        self.key, self.host, self.port = output / "pod_key", host, port
        self.options = ["-i", str(self.key), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes"]
        self.options += ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20"]
        self.options += ["-o", f"UserKnownHostsFile={output / 'known_hosts'}"]
        self.options += ["-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]

    def run(self, command, log, seconds):
        """One command on the host; its status. A command over its limit counts as failed."""
        args = ["ssh", *self.options, "-p", str(self.port), "root@" + self.host, command]
        with log.open("ab") as stream:
            try:
                return subprocess.run(
                    args, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds
                ).returncode
            except subprocess.TimeoutExpired:
                stream.write(b"\n--- over the time limit ---\n")
                return 124

    def copy(self, sources, target, log, seconds, back=False):
        place = f"root@{self.host}:"
        names = [place + s for s in sources] if back else [str(s) for s in sources]
        args = ["scp", "-q", "-r", *self.options, "-P", str(self.port), *names]
        args.append(str(target) if back else place + target)
        with log.open("ab") as stream:
            try:
                return subprocess.run(
                    args, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds
                ).returncode
            except subprocess.TimeoutExpired:
                stream.write(b"\n--- over the time limit ---\n")
                return 124


def dollars(text, what):
    try:
        value = Decimal(text)
    except (InvalidOperation, TypeError):
        raise ValueError(f"invalid {what}") from None
    require(value.is_finite() and value > 0, f"invalid {what}")
    return value


def ceiling(price):
    """The most the host can cost: the hourly price times every time limit, rounded up to a cent."""
    seconds = sum(LIMITS.values())
    return (price * seconds / 3600).quantize(Decimal("0.01"), rounding="ROUND_UP"), seconds


def enrollment(build, gguf):
    """What this caller will accept, from the repository's pins and the caller's own files."""
    kit, case = proof.KIT, proof.CASE
    require(sha(PROBE / "kit.json") == ENROLLMENT["kit_json_sha256"], "kit.json is not enrolled")
    require(
        sha(ROOT / kit["model"]["case"]) == ENROLLMENT["case_sha256"], "the case is not enrolled"
    )
    require(
        ENROLLMENT["verifier_source_digest"] == ft.PINS["verifier_source_digest"],
        "the enrolled verifier is not the patched one",
    )
    require(
        re.fullmatch("[0-9a-f]{64}", ENROLLMENT["commitment_id"]) is not None,
        "the enrolled commitment ID is not 64 hex digits",
    )
    require(gguf.is_file() and not gguf.is_symlink(), "the model file is missing")
    require(gguf.name == kit["gguf"]["name"], "the model file has another name")
    require(sha(gguf) == kit["gguf"]["sha256"], "the model file is not the one that kit.json pins")
    for name in PROGRAMS:
        path = build / "hc" / name
        require(path.is_file() and not path.is_symlink(), f"hc/{name} is missing from the build")
    request = proof.request_bytes()
    return dict(
        schema_id="hallmark.flashprove-tinyllama-run-enrollment/1",
        enrolled=ENROLLMENT,
        kit=dict(files=kit["files"], h_par=kit["h_par"], configuration=kit["configuration"]),
        model=kit["gguf"],
        request=dict(
            sha256=hashlib.sha256(request).hexdigest(),
            prompt_tokens=len(case["prompt_ids"]),
            answer_tokens=len(case["answer_ids"]),
            range=f"{kit['request']['r']}:{kit['request']['e']}",
            end_token=case["answer_ids"][-1],
        ),
        programs={name: sha(build / "hc" / name) for name in PROGRAMS},
        patches=ft.PINS["patches"],
        source=fp.PIN["revision"],
    )


def stage(output, tree, build, gguf):
    """gpu_job.sh's STAGE, from the caller's own files: all of it public."""
    folder = output / "stage"
    for part in ("hc", "config", "weights"):
        (folder / part).mkdir(parents=True)
    for name in PROGRAMS:
        shutil.copyfile(build / "hc" / name, folder / "hc" / name)
        (folder / "hc" / name).chmod(0o755)
    for name in proof.CONFIG:
        shutil.copyfile(PROBE / "kit" / name, folder / "config" / name)
    for name in ("wsums.txt", "fold.txt"):
        shutil.copyfile(PROBE / "kit" / name, folder / "weights" / name)
    if gguf.stat().st_dev == folder.stat().st_dev:
        os.link(gguf, folder / "weights" / gguf.name)  # 2.2 GB: no second copy on one disk
    else:
        shutil.copyfile(gguf, folder / "weights" / gguf.name)
    (folder / "example.req").write_bytes(proof.request_bytes())
    seed = tree / ENROLLMENT["seed"]["path"]
    require(sha(seed) == ENROLLMENT["seed"]["sha256"], "the seed is not upstream's published one")
    shutil.copyfile(seed, folder / "public-seed.bin")
    for name in ("gpu_job.sh", "pod_job.sh"):
        shutil.copyfile(PROBE / name, folder / name)
    pins = {f"config/{name}": proof.KIT["files"][name] for name in proof.CONFIG}
    pins["public-seed.bin"] = ENROLLMENT["seed"]["sha256"]
    pins[f"weights/{gguf.name}"] = proof.KIT["gguf"]["sha256"]
    (folder / "expected.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in pins.items())
    )
    return folder


def pod_spec(public_key, gpu, cloud, deadline):
    start = (
        "bash -c 'apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "
        "openssh-server; mkdir -p ~/.ssh; chmod 700 ~/.ssh; "
        'echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; '
        "service ssh start; sleep infinity'"
    )
    return dict(
        name="hallmark-flashprove-tinyllama",
        cloudType=cloud,
        gpuCount=1,
        gpuTypeId=gpu,
        imageName=POD["image"],
        containerDiskInGb=POD["container_disk_gb"],
        volumeInGb=0,
        minVcpuCount=POD["min_vcpu"],
        minMemoryInGb=POD["min_memory_gb"],
        ports="22/tcp",
        supportPublicIp=True,
        startSsh=True,
        dockerArgs=start,
        env=[
            dict(key="PUBLIC_KEY", value=public_key),
            dict(key="NVIDIA_DRIVER_CAPABILITIES", value="compute,utility"),
        ],
        # The provider's own end of the host, should both of this caller's processes be gone.
        terminateAfter=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline + POD["grace"])),
    )


def end_host(provider, pod_id, attempts=5, wait=10):
    """Delete the host and see that the account no longer lists it."""
    for _ in range(attempts):
        try:
            provider.terminate(pod_id)
        except (ValueError, OSError):
            pass
        try:
            if pod_id not in provider.account()[1]:
                return True
        except (ValueError, OSError):
            pass
        time.sleep(wait)
    return False


def watchdog(pod_id, deadline, log):
    """A second process: at the deadline it deletes the host, whatever became of the first."""
    provider = RunPod(os.environ.get("RUNPOD_API_KEY"))
    while time.time() < deadline:
        time.sleep(5)
    ended = end_host(provider, pod_id)
    with open(log, "a") as stream:
        stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ended={ended}\n")


def start_guard(pod_id, deadline, output):
    return subprocess.Popen(
        [sys.executable, __file__, "--watchdog", pod_id, str(deadline), str(output / "guard.log")],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def step(code, name):
    require(code == 0, f"the host's step {name} ended with status {code}")


def on_host(provider, remote_of, output, staged, proofs, pod_id, started):
    """Every paid step, each under its limit; returns the folder with what came back."""
    boot = started + LIMITS["boot"]
    address = None
    while address is None:
        require(time.time() < boot, "the host did not start in time")
        time.sleep(10)
        address = provider.address(pod_id)
    remote = remote_of(output, *address)
    log = output / "host.log"
    while remote.run("true", log, 30) != 0:
        require(time.time() < boot, "the host's SSH service did not start in time")
        time.sleep(10)
    step(remote.copy([PROBE / "pod_job.sh"], "/root/pod_job.sh", log, 60), "copy")
    step(remote.run("bash /root/pod_job.sh host", log, LIMITS["host"]), "host")
    step(remote.copy([staged], BASE + "/stage", log, LIMITS["upload"]), "upload")
    step(remote.run("bash /root/pod_job.sh libraries", log, LIMITS["libraries"]), "libraries")
    proving = f"runuser -u fp -- bash {BASE}/stage/pod_job.sh prove {proofs}"
    code = remote.run(proving, output / "prove.log", LIMITS["prove"])
    back = output / "prove"
    back.mkdir()
    work = BASE + "/run/work/"
    names = [work + name for name in RETURNED] + [work + "public"]
    numbers = range(1, proofs + 1)
    names += [work + f"cp_prove.{n}.{kind}" for n in numbers for kind in ("log", "time")]
    names += [work + f"cp_verify.{n}.log" for n in numbers]
    fetched = remote.copy(names, back, log, LIMITS["fetch"], back=True)
    step(code, "prove")
    step(fetched, "fetch")
    return back


def judged(back, staged, record, local, built):
    """The caller's own checks of what came back; returns the local check's result."""
    commitment_id = ENROLLMENT["commitment_id"]
    printed = (back / "commitment-id.txt").read_text().strip()
    require(printed == commitment_id, "cp_commit printed another commitment ID than the enrolled")
    binaries = dict(
        reversed(line.split("  ")) for line in (back / "binaries.sha256").read_text().splitlines()
    )
    require(binaries == record["programs"], "the host ran other programs than the caller's")
    for line in (back / "inputs.sha256").read_text().splitlines():
        digest, name = line.split("  ")
        require(sha(staged / name) == digest, f"the host read another {name} than the caller's")
    return proof.verify(back / "public", commitment_id, local, built)


def run(balance, maximum, build, gguf, proofs, gpu, cloud, provider=None, remote_of=Remote):
    cap, stated = dollars(maximum, "cost cap"), dollars(balance, "credit balance")
    require(1 <= proofs <= 9, "one to nine proofs")
    require(gpu in POD["gpus"] and cloud in ("SECURE", "COMMUNITY"), "no such GPU or cloud here")
    RUNS.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="flashprove-tinyllama-gpu-", dir=RUNS))
    print("Outputs: " + str(output), flush=True)
    began = time.monotonic()
    pod_id, spend = None, None
    try:
        # Free steps first: what the caller accepts, and the verifiers that will judge the proof.
        record = enrollment(build, gguf)
        # The request's ids are the pinned tokenizer's and chat template's for the case's text.
        record["request"]["text"] = request_text.check_text(output)
        write_json(output / "enrollment.json", record)
        local = output / "local"
        local.mkdir()
        built = proof.built_verifiers(local)  # the verifiers that will judge the paid proof
        tree = built[0]
        staged = stage(output, tree, build, gguf)
        provider = provider or RunPod(os.environ.get("RUNPOD_API_KEY"))
        price = provider.price(gpu, cloud)
        most, seconds = ceiling(price)
        require(most <= cap, f"the worst case, USD {most}, is over the cap")
        before, pods = provider.account()
        require(not pods, "another host of this account exists")
        require(most <= stated and most <= before, "the balance does not cover the worst case")
        write_json(
            output / "protocol.json",
            dict(
                schema_id="hallmark.flashprove-tinyllama-gpu/1",
                source_files={
                    str(p.relative_to(ROOT)): sha(p)
                    for p in (Path(__file__), PROBE / "gpu_job.sh", PROBE / "pod_job.sh")
                },
                pod=POD,
                gpu=gpu,
                cloud=cloud,
                proofs=proofs,
                price_usd_per_hour=str(price),
                seconds_limit=seconds,
                cost_ceiling_usd=str(most),
                cost_cap_usd=str(cap),
                cap_set_by="the caller of this run",
                owner_approved=False,
                balance_check_usd=str(stated),
                automatic_retries=0,
                public_weights=True,
                hiding_claim=False,
                qualified_backend=False,
            ),
        )
        key = output / "pod_key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "", "-f", str(key)], check=True
        )
        public_key = (output / "pod_key.pub").read_text().strip()
        started = time.time()
        deadline = started + seconds
        try:
            pod = provider.create(pod_spec(public_key, gpu, cloud, deadline))
        except BaseException:
            # No host existed before: one that exists now was made by the failed call.
            for other in provider.account()[1]:
                end_host(provider, other)
            raise
        pod_id, guard = pod["id"], None
        try:
            (output / "pod-id.txt").write_text(pod_id + "\n")  # to delete it by hand at need
            guard = start_guard(pod_id, deadline, output)
            require(Decimal(str(pod["costPerHr"])) <= price, "the host costs more than listed")
            back = on_host(provider, remote_of, output, staged, proofs, pod_id, started)
        finally:
            ended = end_host(provider, pod_id)
            paid = time.time() - started
            try:
                after = provider.account()[0] if ended else None
            except (ValueError, OSError):
                after = None
            spend = dict(
                host_seconds=round(paid, 1),
                bound_usd=str((price * Decimal(paid) / 3600).quantize(Decimal("0.0001"))),
                balance_fall_usd=None if after is None else str(before - after),
                cap_usd=str(cap),
                within_cap=price * Decimal(paid) / 3600 <= cap,
                host_deleted=ended,
            )
            write_json(output / "spend.json", spend)
            if ended and guard is not None:
                guard.terminate()
            require(ended, "THE HOST MAY STILL EXIST: delete it in the provider's console")
        host = {k: pod[k] for k in ("gpuCount", "vcpuCount", "memoryInGb")}
        host["gpu"] = (pod.get("machine") or {}).get("gpuDisplayName")
        result = judged(back, staged, record, local, built)
        result |= dict(
            passed=True,
            schema_id="hallmark.flashprove-tinyllama-gpu-result/1",
            enrollment_sha256=sha(output / "enrollment.json"),
            worker_verdicts=[
                (back / f"cp_verify.{n}.log").read_text().strip().splitlines()[-1]
                for n in range(1, proofs + 1)
            ],
            proofs=proofs,
            host=host,
            spend=spend,
            seconds_in_all=time.monotonic() - began,
            invoice=False,
            proof_made_here=False,
            gpu_used_here=False,
            hiding_claim=False,
            qualified_backend=False,
        )
        write_json(output / "result.json", result)
        print(
            json.dumps(
                {k: result[k] for k in ("passed", "patched_verdict", "altered_copies_rejected")}
                | dict(spend_bound_usd=spend["bound_usd"])
            )
        )
        return output
    except BaseException as exc:
        write_json(
            output / "failure.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                spend=spend,
                host_made=pod_id is not None,
            ),
        )
        raise


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--watchdog":
        watchdog(sys.argv[2], float(sys.argv[3]), sys.argv[4])
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--credit-balance-usd", required=True)
    p.add_argument("--max-cost-usd", required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--proofs", type=int, default=1)
    p.add_argument("--gpu", default=POD["gpus"][0])
    p.add_argument("--cloud", default="SECURE")
    a = p.parse_args()

    def stop(number, _frame):
        raise KeyboardInterrupt(f"signal {number}")

    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop)
    run(
        a.credit_balance_usd,
        a.max_cost_usd,
        a.build.resolve(),
        a.gguf.resolve(),
        a.proofs,
        a.gpu,
        a.cloud,
    )


if __name__ == "__main__":
    main()
