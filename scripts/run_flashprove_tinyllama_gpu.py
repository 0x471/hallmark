"""Commit to and prove dense TinyLlama on one rented sm_90 GPU under a spend cap; recheck here.

usage: python scripts/run_flashprove_tinyllama_gpu.py --credit-balance-usd USD --max-cost-usd USD
       --build DIR --gguf FILE [--proofs N] [--gpu NAME] [--cloud SECURE|COMMUNITY]

One paid host, one attempt, no retry. Before anything is paid for, the caller's enrollment is
written (enrollment.json: the kit, the request, the commitment ID and the patched verifier's
digest that this caller will accept), the pinned tokenizer and chat template must give the
request's ids for the case's text, the patched and the unpatched cp_verify are built here from
the pinned source, and the planned worst case of the charge must fit the cap, the balance that
the caller states and the balance that the provider reports. The host runs gpu_job.sh on public
files only; the provider's secret directory never leaves it. Then
scripts/flashprove_tinyllama_proof.py's checks run here on the returned files, under the enrolled
commitment ID: a commitment with another ID fails the run.

How the host ends. This process deletes it after the last step or the first failure and sees
that the account no longer lists it. A second process is started before the host is asked for;
it deletes every host of this run's name once this process is gone or the deadline has passed,
and keeps trying for half an hour. Both run on the caller's machine: if that machine loses its
network or sleeps, neither can act, and the host is billed until it is deleted by hand. The
provider is asked to end the host itself (terminateAfter), but did not in the one drill made.
The planned worst case is the hourly price times every step's limit and the time allowed for a
deletion that succeeds; a deletion that fails costs more than that.

DIR holds the patched programs hc/cp_commit, hc/cp_prove and hc/cp_verify (upstream's
prover/build.sh and verifier/build.sh on the pinned source with the three program patches
applied; this script does not build them). FILE is the exported model, the one that kit.json
pins (the answer target rebuilds it). The provider's API key is read from the environment
(RUNPOD_API_KEY) at the start, taken out of it, and given to the second process alone; it is
never written anywhere.

The cap is the caller's own: this script makes no claim that an owner approved it. Its spend
figure is a bound from the provider's hourly price and this process's clock, from just before the
host is asked for until it is seen deleted. The fall of the account's balance is read once beside
it, but the provider charges late and in steps. Neither is an invoice.
"""

import argparse
import hashlib
import http.client
import json
import os
import re
import secrets
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
    """The calls this controller makes of the provider. The key stays in this object."""

    URL = "https://api.runpod.io/graphql"

    def __init__(self, key):
        require(bool(key), "RUNPOD_API_KEY is not set")
        self._key = key

    def _call(self, query, variables=None, timeout=60):
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
            with urllib.request.urlopen(request, timeout=timeout) as reply:
                value = json.loads(reply.read().decode())
        except urllib.error.HTTPError as error:
            raise ValueError(f"the provider answered {error.code}") from None
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            raise ValueError("the provider could not be reached: " + type(error).__name__) from None
        if not isinstance(value, dict) or "errors" in value or not value.get("data"):
            said = ""
            if isinstance(value, dict) and value.get("errors"):
                said = str(value["errors"][0].get("message", ""))[:300]
            raise ValueError("the provider refused the call: " + said)
        return value["data"]

    def account(self, timeout=60):
        """(balance in USD, {id: name} of the hosts that exist)."""
        me = self._call("query { myself { clientBalance pods { id name } } }", timeout=timeout)
        me = me.get("myself")
        require(
            isinstance(me, dict) and isinstance(me.get("pods"), list), "no account in the reply"
        )
        return Decimal(str(me["clientBalance"])), {pod["id"]: pod["name"] for pod in me["pods"]}

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

    def terminate(self, pod_id, timeout=60):
        query = "mutation($id: String!) { podTerminate(input: {podId: $id}) }"
        self._call(query, {"id": pod_id}, timeout=timeout)


class Remote:
    """SSH and scp to the rented host with a key made for this run alone. The caller's own SSH
    configuration and agent are not used: the host is not trusted."""

    def __init__(self, output, host, port):
        self.key, self.host, self.port = output / "pod_key", host, port
        self.options = ["-F", "/dev/null", "-i", str(self.key), "-o", "IdentitiesOnly=yes"]
        self.options += ["-o", "BatchMode=yes", "-o", "ForwardAgent=no", "-o", "ForwardX11=no"]
        self.options += ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=20"]
        self.options += ["-o", f"UserKnownHostsFile={output / 'known_hosts'}"]
        self.options += ["-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]

    def _run(self, args, log, seconds):
        with log.open("ab") as stream:
            try:
                return subprocess.run(
                    args, stdout=stream, stderr=subprocess.STDOUT, timeout=seconds
                ).returncode
            except subprocess.TimeoutExpired:
                stream.write(b"\n--- over the time limit ---\n")
                return 124

    def run(self, command, log, seconds):
        """One command on the host; its status. A command over its limit counts as failed."""
        args = ["ssh", *self.options, "-p", str(self.port), "root@" + self.host, command]
        return self._run(args, log, seconds)

    def copy(self, sources, target, log, seconds, back=False):
        place = f"root@{self.host}:"
        names = [place + s for s in sources] if back else [str(s) for s in sources]
        args = ["scp", "-q", "-r", *self.options, "-P", str(self.port), *names]
        args.append(str(target) if back else place + target)
        return self._run(args, log, seconds)


def dollars(text, what):
    try:
        value = Decimal(text)
    except (InvalidOperation, TypeError):
        raise ValueError(f"invalid {what}") from None
    require(value.is_finite() and value > 0, f"invalid {what}")
    return value


def ceiling(price):
    """The planned worst case, rounded up to a cent, and the seconds its steps may take: the
    hourly price times every step's limit and the time allowed for a deletion that succeeds
    (end_host() gives up after that time). A host that cannot be deleted costs more."""
    seconds = sum(LIMITS.values())
    most = price * (seconds + POD["end_seconds"]) / 3600
    return most.quantize(Decimal("0.01"), rounding="ROUND_UP"), seconds


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


def pod_spec(public_key, name, gpu, cloud, deadline):
    # A failed step ends the start command, so that the host never looks ready without SSH.
    start = (
        "bash -c 'set -e; apt-get update -qq; DEBIAN_FRONTEND=noninteractive apt-get install -y "
        "-qq openssh-server; mkdir -p ~/.ssh; chmod 700 ~/.ssh; "
        'echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys; '
        "service ssh start; sleep infinity'"
    )
    return dict(
        name=name,
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
        # Asked of the provider for the case that both of this caller's processes are gone. In a
        # drill of 2026-11-27 the provider left a host running past this time: not relied upon.
        terminateAfter=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(deadline + POD["grace"])),
    )


def end_hosts(provider, name, seconds):
    """Delete every host of this run's name and see that the account lists none, trying for at
    most `seconds`. True once none is listed. No error of the provider or the network ends it."""
    until = time.time() + seconds
    while True:
        try:
            mine = [i for i, n in provider.account(timeout=10)[1].items() if n == name]
            if not mine:
                return True
            for pod_id in mine:
                provider.terminate(pod_id, timeout=10)
        except Exception:  # any failure is tried again until the time is up
            pass
        if time.time() + 5 >= until:
            return False
        time.sleep(5)


def watchdog(name, deadline, parent, log):
    """The second process. It waits until the first is gone or the deadline has passed, then
    deletes this run's hosts and keeps trying for POD["guard_seconds"]."""
    provider = RunPod(os.environ.pop("RUNPOD_API_KEY", None))
    while time.time() < deadline and os.getppid() == parent:
        time.sleep(2)
    ended = end_hosts(provider, name, POD["guard_seconds"])
    with open(log, "a") as stream:
        stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} ended={ended}\n")


def start_guard(name, deadline, output, key):
    """Starts the second process, with the key in its environment alone."""
    args = [sys.executable, __file__, "--watchdog", name, str(deadline), str(os.getpid())]
    return subprocess.Popen(
        [*args, str(output / "guard.log")],
        env=os.environ | {"RUNPOD_API_KEY": key},
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def balance_fall(provider, before):
    """The fall of the account's balance since `before`, read once after the host is deleted, or
    None. The provider charges in steps and late: this can miss a part of this host's charge and
    can hold an earlier host's. The bound from the price and the clock is the figure to go by."""
    try:
        return (before - provider.account()[0]).quantize(Decimal("0.0001"))
    except Exception:  # the reading is an aside
        return None


def step(code, name):
    require(code == 0, f"the host's step {name} ended with status {code}")


def on_host(provider, remote_of, output, staged, proofs, pod_id, started):
    """Every paid step, each under its limit and all of them under the deadline; returns the
    folder with what came back."""
    deadline = started + sum(LIMITS.values())

    def within(name):
        left = int(min(LIMITS[name], deadline - time.time()))
        require(left > 0, f"no time is left for the host's step {name}")
        return left

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
    step(remote.copy([PROBE / "pod_job.sh"], "/root/pod_job.sh", log, within("host")), "copy")
    step(remote.run("bash /root/pod_job.sh host", log, within("host")), "host")
    step(remote.copy([staged], BASE + "/stage", log, within("upload")), "upload")
    step(remote.run("bash /root/pod_job.sh libraries", log, within("libraries")), "libraries")
    proving = f"runuser -u fp -- bash {BASE}/stage/pod_job.sh prove {proofs}"
    code = remote.run(proving, output / "prove.log", within("prove"))
    back = output / "prove"
    back.mkdir()
    work = BASE + "/run/work/"
    names = [work + name for name in RETURNED] + [work + "public", BASE + "/cuda-packages.json"]
    numbers = range(1, proofs + 1)
    names += [work + f"cp_prove.{n}.{kind}" for n in numbers for kind in ("log", "time")]
    names += [work + f"cp_verify.{n}.log" for n in numbers]
    fetched = remote.copy(names, back, log, within("fetch"), back=True)
    step(code, "prove")
    step(fetched, "fetch")
    return back


def sums(path):
    """{name: digest} of a sha256sum file; a line of another form fails."""
    rows = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        require(match is not None and match[2] not in rows, f"{path.name} has an unexpected line")
        rows[match[2]] = match[1]
    return rows


def judged(back, staged, record, local, built, proofs):
    """The caller's own checks of what came back; returns the local check's result."""
    commitment_id = ENROLLMENT["commitment_id"]
    printed = (back / "commitment-id.txt").read_text().strip()
    require(printed == commitment_id, "cp_commit printed another commitment ID than the enrolled")
    require(
        sums(back / "binaries.sha256") == record["programs"],
        "the host ran other programs than the caller's",
    )
    inputs = sums(back / "inputs.sha256")
    wanted = {f"config/{name}" for name in proof.CONFIG} | {"weights/wsums.txt", "weights/fold.txt"}
    wanted |= {"example.req", "public-seed.bin", "weights/" + proof.KIT["gguf"]["name"]}
    require(set(inputs) == wanted, "the host's record of its inputs names other files")
    for name, digest in inputs.items():
        require(sha(staged / name) == digest, f"the host read another {name} than the caller's")
    for n in range(1, proofs + 1):
        last = (back / f"cp_verify.{n}.log").read_text().strip().splitlines()[-1:]
        require(last == ["YES"], f"the host's cp_verify did not accept proof {n}")
    return proof.verify(back / "public", commitment_id, local, built)


def run(key, balance, maximum, build, gguf, proofs, gpu, cloud, provider=None, remote_of=Remote):
    cap, stated = dollars(maximum, "cost cap"), dollars(balance, "credit balance")
    require(1 <= proofs <= 9, "one to nine proofs")
    require(gpu in POD["gpus"] and cloud in ("SECURE", "COMMUNITY"), "no such GPU or cloud here")
    provider = provider or RunPod(key)  # a missing key fails here, before any long step
    RUNS.mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="flashprove-tinyllama-gpu-", dir=RUNS))
    print("Outputs: " + str(output), flush=True)
    began = time.monotonic()
    asked, spend = False, None
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
        price = provider.price(gpu, cloud)
        most, seconds = ceiling(price)
        require(most <= cap, f"the planned worst case, USD {most}, is over the cap")
        before, pods = provider.account()
        require(not pods, "another host of this account exists")
        require(most <= stated and most <= before, "the balance does not cover the worst case")
        write_json(
            output / "protocol.json",
            dict(
                schema_id="hallmark.flashprove-tinyllama-gpu/2",
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
                planned_worst_case_usd=str(most),
                planned_worst_case_note="if the host can be deleted within pod.end_seconds",
                cost_cap_usd=str(cap),
                cap_set_by="the caller of this run",
                owner_approved=False,
                balances_cover_the_worst_case=True,
                automatic_retries=0,
                public_weights=True,
                hiding_claim=False,
                qualified_backend=False,
            ),
        )
        key_file = output / "pod_key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "", "-f", str(key_file)],
            check=True,
        )
        public_key = (output / "pod_key.pub").read_text().strip()
        name = "hallmark-fp-" + secrets.token_hex(6)  # this run's hosts, and no other's
        started = time.time()
        deadline = started + seconds
        guard = start_guard(name, deadline, output, key)  # before any host can exist
        ended, back, pod = False, None, None
        asked = True
        try:
            time.sleep(1)
            require(guard.poll() is None, "the second process did not start")
            pod = provider.create(pod_spec(public_key, name, gpu, cloud, deadline))
            (output / "pod-id.txt").write_text(pod["id"] + "\n")  # to delete it by hand at need
            require(Decimal(str(pod["costPerHr"])) <= price, "the host costs more than listed")
            back = on_host(provider, remote_of, output, staged, proofs, pod["id"], started)
        finally:
            # Nothing may interrupt the deletion and its record.
            numbers = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
            handlers = [signal.signal(number, signal.SIG_IGN) for number in numbers]
            # After a failed call a host may be listed late: look for longer then.
            ended = end_hosts(provider, name, POD["end_seconds"])
            if ended and pod is None:
                time.sleep(POD["late_seconds"])
                ended = end_hosts(provider, name, POD["end_seconds"])
            paid = time.time() - started
            fall = balance_fall(provider, before) if ended else None
            bound = (price * Decimal(paid) / 3600).quantize(Decimal("0.0001"), rounding="ROUND_UP")
            spend = dict(
                host_seconds=round(paid, 1),
                price_usd_per_hour=str(price),
                bound_usd=str(bound),
                balance_fall_at_end_usd=None if fall is None else str(fall),
                balance_fall_note="read once at the end; the provider charges late and in steps",
                cap_usd=str(cap),
                within_cap=bound <= cap,
                host_deleted=ended,
            )
            write_json(output / "spend.json", spend)
            if ended:
                guard.terminate()
            else:
                print(
                    "THE HOST MAY STILL EXIST AND IS BILLED. The second process keeps trying; "
                    "check the provider's console and delete it there.",
                    flush=True,
                )
            for number, handler in zip(numbers, handlers, strict=True):
                signal.signal(number, handler)
        require(ended, "THE HOST MAY STILL EXIST: delete it in the provider's console")
        host = {k: pod[k] for k in ("gpuCount", "vcpuCount", "memoryInGb")}
        host["gpu"] = (pod.get("machine") or {}).get("gpuDisplayName")
        result = judged(back, staged, record, local, built, proofs)
        result |= dict(
            passed=True,
            schema_id="hallmark.flashprove-tinyllama-gpu-result/2",
            enrollment_sha256=sha(output / "enrollment.json"),
            proofs_made_on_the_host=proofs,
            proofs_checked_here=1,
            worker_verdicts=["YES"] * proofs,
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
                host_asked_for=asked,
            ),
        )
        raise


def main():
    if len(sys.argv) == 6 and sys.argv[1] == "--watchdog":
        watchdog(sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5])
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
    # The key leaves the environment before any child of this process is started.
    key = os.environ.pop("RUNPOD_API_KEY", None)

    def stop(number, _frame):
        raise KeyboardInterrupt(f"signal {number}")

    for number in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(number, stop)
    run(
        key,
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
