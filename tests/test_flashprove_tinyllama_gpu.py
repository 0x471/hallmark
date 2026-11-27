"""The paid dense TinyLlama run's controller on a stand-in provider and host; nothing is paid."""

import json
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_flashprove_tinyllama_gpu as gpu

ID = gpu.ENROLLMENT["commitment_id"]
H100 = gpu.POD["gpus"][0]


class Provider:
    """Lists one host from create() until terminate(); records every call."""

    def __init__(self, balance="50.00", price="4.39", pods=(), cost=4.39, deletes=True):
        self.balance, self.listed, self.pods = Decimal(balance), Decimal(price), list(pods)
        self.cost, self.deletes, self.calls = cost, deletes, []

    def account(self):
        self.calls.append("account")
        return self.balance, list(self.pods)

    def price(self, gpu_name, cloud):
        self.calls.append(("price", gpu_name, cloud))
        return self.listed

    def create(self, spec):
        self.calls.append(("create", spec))
        self.pods.append("pod1")
        return dict(
            id="pod1", costPerHr=self.cost, gpuCount=1, vcpuCount=29, memoryInGb=287, machine=None
        )

    def address(self, pod_id):
        return "198.51.100.7", 2222

    def terminate(self, pod_id):
        self.calls.append(("terminate", pod_id))
        if self.deletes:
            self.pods.remove(pod_id) if pod_id in self.pods else None
            self.balance -= Decimal("0.25")


def host(fail=None, commitment_id=ID, programs=None, verdict="YES"):
    """A stand-in for the rented host: answers the controller's commands and copies."""

    class Host:
        commands = []

        def __init__(self, output, address, port):
            self.output = output

        def run(self, command, log, seconds):
            Host.commands.append(command)
            return 0 if fail is None or fail not in command else 9

        def copy(self, sources, target, log, seconds, back=False):
            if not back:
                return 0
            record = json.loads((self.output / "enrollment.json").read_text())
            sums = programs or record["programs"]
            staged = self.output / "stage"
            (target / "public").mkdir()
            (target / "commitment-id.txt").write_text(commitment_id + "\n")
            (target / "binaries.sha256").write_text(
                "".join(f"{digest}  {name}\n" for name, digest in sums.items())
            )
            (target / "inputs.sha256").write_text(
                f"{gpu.sha(staged / 'example.req')}  example.req\n"
            )
            (target / "cp_verify.1.log").write_text(f"result: ...\n{verdict}\n")
            return 0

    Host.commands = []
    return Host


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    """A build, a model file and a source tree that stand in for the 2.2 GB of the real ones."""
    build, tree = tmp_path / "build", tmp_path / "tree"
    (build / "hc").mkdir(parents=True)
    for name in gpu.PROGRAMS:
        (build / "hc" / name).write_bytes(name.encode())
    gguf = tmp_path / gpu.proof.KIT["gguf"]["name"]
    gguf.write_bytes(b"weights")
    monkeypatch.setitem(gpu.proof.KIT["gguf"], "sha256", gpu.sha(gguf))
    seed = tree / gpu.ENROLLMENT["seed"]["path"]
    seed.parent.mkdir(parents=True)
    seed.write_bytes(b"s" * 32)
    monkeypatch.setitem(gpu.ENROLLMENT["seed"], "sha256", gpu.sha(seed))
    runs = tmp_path / "runs"
    monkeypatch.setattr(gpu, "RUNS", runs)
    monkeypatch.setattr(gpu.proof, "built_verifiers", lambda output: (tree, {}, {}))
    monkeypatch.setattr(gpu.request_text, "check_text", lambda output: dict(prompt_tokens=25))
    monkeypatch.setattr(gpu.time, "sleep", lambda seconds: None)
    started = []

    class Guard:
        def __init__(self, pod_id, deadline, output):
            started.append(pod_id)

        def terminate(self):
            started.append("stopped")

    monkeypatch.setattr(gpu, "start_guard", Guard)
    checked = dict(patched_verdict="YES", altered_copies_rejected=8)
    monkeypatch.setattr(gpu.proof, "verify", lambda case, cid, output, built: dict(checked))
    return dict(build=build, gguf=gguf, runs=runs, guard=started)


def go(inputs, provider, remote, balance="50.00", cap="3.00", proofs=1, cloud="SECURE"):
    return gpu.run(
        balance, cap, inputs["build"], inputs["gguf"], proofs, H100, cloud, provider, remote
    )


def only_run(inputs):
    (folder,) = inputs["runs"].iterdir()
    return folder


def test_the_worst_case_is_the_price_times_every_limit():
    most, seconds = gpu.ceiling(Decimal("4.39"))
    assert seconds == sum(gpu.LIMITS.values()) == 2400 and gpu.POD["end_seconds"] == 60
    assert most == Decimal("3.00")  # 4.39 * 2460 / 3600 = 2.9998, rounded up
    assert gpu.ceiling(Decimal("4.40"))[0] == Decimal("3.01")


@pytest.mark.parametrize("text", ["abc", "0", "-1", "nan", "inf", None])
def test_an_amount_that_is_no_positive_number_is_refused(text):
    with pytest.raises(ValueError, match="invalid"):
        gpu.dollars(text, "cost cap")


def test_the_host_is_asked_to_end_itself_after_the_deadline():
    spec = gpu.pod_spec("ssh-ed25519 AAAA", H100, "SECURE", 1_800_000_000)
    after = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1_800_000_000 + gpu.POD["grace"]))
    assert spec["terminateAfter"] == after and spec["gpuCount"] == 1 and spec["volumeInGb"] == 0
    assert {e["key"] for e in spec["env"]} == {"PUBLIC_KEY", "NVIDIA_DRIVER_CAPABILITIES"}


def test_a_passing_run_ends_the_host_and_records_the_spend(inputs):
    provider, remote = Provider(), host()
    output = go(inputs, provider, remote)
    result = json.loads((output / "result.json").read_text())
    spend = json.loads((output / "spend.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    assert result["passed"] is True and result["worker_verdicts"] == ["YES"]
    assert result["spend"] == spend and spend["host_deleted"] and spend["within_cap"]
    assert spend["balance_fall_usd"] == "0.2500" and spend["cap_usd"] == "3.00"
    assert spend["price_usd_per_hour"] == "4.39"
    assert protocol["cost_ceiling_usd"] == "3.00" and protocol["owner_approved"] is False
    assert ("terminate", "pod1") in provider.calls and provider.pods == []
    assert inputs["guard"][-1] == "stopped"
    # The enrollment is written before the host is made, and nothing names the host in the result.
    names = [c[0] if isinstance(c, tuple) else c for c in provider.calls]
    assert names.index("create") > names.index("price")
    assert "pod1" not in (output / "result.json").read_text()
    assert remote.commands[-1].endswith("pod_job.sh prove 1")


@pytest.mark.parametrize(
    ("provider", "cap", "balance", "why"),
    [
        (Provider(), "2.99", "50.00", "over the cap"),
        (Provider(pods=["other"]), "3.00", "50.00", "another host"),
        (Provider(), "3.00", "2.99", "does not cover"),
        (Provider(balance="2.99"), "3.00", "50.00", "does not cover"),
        (Provider(price="5.29"), "3.00", "50.00", "over the cap"),
    ],
)
def test_no_host_is_made_when_the_worst_case_does_not_fit(inputs, provider, cap, balance, why):
    with pytest.raises(ValueError, match=why):
        go(inputs, provider, host(), balance=balance, cap=cap)
    assert not any(isinstance(c, tuple) and c[0] == "create" for c in provider.calls)
    failure = json.loads((only_run(inputs) / "failure.json").read_text())
    assert failure["host_made"] is False and failure["spend"] is None


@pytest.mark.parametrize(
    "change",
    [
        lambda i: i["gguf"].write_bytes(b"other weights"),
        lambda i: (i["build"] / "hc/cp_prove").unlink(),
        lambda i: i["gguf"].rename(i["gguf"].with_name("other.gguf")),
    ],
)
def test_no_host_is_made_for_files_that_are_not_the_enrolled_ones(inputs, change):
    change(inputs)
    gguf = inputs["gguf"] if inputs["gguf"].exists() else inputs["gguf"].with_name("other.gguf")
    provider = Provider()
    with pytest.raises(ValueError):
        gpu.run("50.00", "3.00", inputs["build"], gguf, 1, H100, "SECURE", provider, host())
    assert provider.calls == []


@pytest.mark.parametrize("step", ["pod_job.sh host", "pod_job.sh libraries", "pod_job.sh prove"])
def test_a_failed_step_still_ends_the_host(inputs, step):
    provider = Provider()
    with pytest.raises(ValueError, match="ended with status 9"):
        go(inputs, provider, host(fail=step))
    assert ("terminate", "pod1") in provider.calls and provider.pods == []
    failure = json.loads((only_run(inputs) / "failure.json").read_text())
    assert failure["host_made"] is True and failure["spend"]["host_deleted"] is True
    assert not (only_run(inputs) / "result.json").exists()


def test_a_host_that_costs_more_than_listed_is_ended_at_once(inputs):
    provider, remote = Provider(cost=4.40), host()
    with pytest.raises(ValueError, match="costs more"):
        go(inputs, provider, remote)
    assert ("terminate", "pod1") in provider.calls and remote.commands == []


def test_a_host_that_cannot_be_deleted_is_said_loudly_and_the_guard_stays(inputs):
    provider = Provider(deletes=False)
    with pytest.raises(ValueError, match="THE HOST MAY STILL EXIST"):
        go(inputs, provider, host())
    assert "stopped" not in inputs["guard"]
    assert json.loads((only_run(inputs) / "spend.json").read_text())["host_deleted"] is False


@pytest.mark.parametrize(
    ("remote", "why"),
    [
        (host(commitment_id="0" * 64), "another commitment ID"),
        (host(programs={"cp_commit": "0" * 64}), "other programs"),
    ],
)
def test_what_the_host_returns_is_held_to_the_enrollment(inputs, remote, why):
    provider = Provider()
    with pytest.raises(ValueError, match=why):
        go(inputs, provider, remote)
    assert provider.pods == [] and not (only_run(inputs) / "result.json").exists()


def test_ending_a_host_retries_until_the_account_no_longer_lists_it(monkeypatch):
    monkeypatch.setattr(gpu.time, "sleep", lambda seconds: None)

    class Slow(Provider):
        def terminate(self, pod_id):
            self.calls.append(("terminate", pod_id))
            if len([c for c in self.calls if c == ("terminate", pod_id)]) == 3:
                self.pods.remove(pod_id)
            else:
                raise ValueError("the provider refused the call")

    slow = Slow(pods=["pod1"])
    assert gpu.end_host(slow, "pod1") is True
    assert slow.calls.count(("terminate", "pod1")) == 3
    never = Provider(pods=["pod1"], deletes=False)
    assert gpu.end_host(never, "pod1", attempts=4) is False
    assert never.calls.count(("terminate", "pod1")) == 4


def test_the_staged_files_are_the_callers_and_the_pins_name_them(inputs):
    go(inputs, Provider(), host())
    staged = only_run(inputs) / "stage"
    pins = dict(
        reversed(line.split("  ")) for line in (staged / "expected.sha256").read_text().splitlines()
    )
    assert set(pins) == {
        "config/policy.txt",
        "config/KL.bin",
        "config/layout.txt",
        "public-seed.bin",
        "weights/" + inputs["gguf"].name,
    }
    assert all(gpu.sha(staged / name) == digest for name, digest in pins.items())
    assert (staged / "example.req").read_bytes() == gpu.proof.request_bytes()
    assert (staged / "gpu_job.sh").read_bytes() == (gpu.PROBE / "gpu_job.sh").read_bytes()


def test_a_host_left_by_a_failed_create_is_ended(inputs):
    class Broken(Provider):
        def create(self, spec):
            self.pods.append("left")
            raise ValueError("the provider refused the call: timeout")

    provider = Broken()
    with pytest.raises(ValueError, match="timeout"):
        go(inputs, provider, host())
    assert ("terminate", "left") in provider.calls and provider.pods == []


def test_the_balance_is_read_until_the_charge_shows(monkeypatch):
    monkeypatch.setattr(gpu.time, "sleep", lambda seconds: None)

    class Late(Provider):
        def account(self):
            self.calls.append("account")
            return (self.balance - (1 if len(self.calls) >= 3 else 0)), []

    late = Late(balance="10")
    assert gpu.settled_balance(late, Decimal("10")) == Decimal("9") and len(late.calls) == 3
    never = Provider(balance="10")
    assert gpu.settled_balance(never, Decimal("10"), tries=4) == Decimal("10")
    assert len(never.calls) == 4


def test_a_step_gets_no_more_time_than_the_deadline_leaves(inputs, monkeypatch):
    remote = host()
    # The host starts at once; 2,395 s later five seconds are left for the step "host".
    clock = iter([1000.0, 1000.0, 3395.0])
    monkeypatch.setattr(gpu.time, "time", lambda: next(clock, 3401.0))
    limits = []
    plain = remote.run

    def timed(self, command, log, seconds):
        limits.append((command, seconds))
        return plain(self, command, log, seconds)

    remote.run = timed
    provider = Provider()
    with pytest.raises(ValueError, match="no time is left for the host's step upload"):
        go(inputs, provider, remote)
    assert limits == [("true", 30), ("bash /root/pod_job.sh host", 5)]
    assert provider.pods == [] and ("terminate", "pod1") in provider.calls
