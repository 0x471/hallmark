"""The paid dense TinyLlama run's controller on a stand-in provider and host; nothing is paid."""

import json
import signal
import sys
import time
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_flashprove_tinyllama_gpu as gpu

ID = gpu.ENROLLMENT["commitment_id"]
H100 = gpu.POD["gpus"][0]
INPUTS = (
    "config/policy.txt",
    "config/KL.bin",
    "config/layout.txt",
    "weights/wsums.txt",
    "weights/fold.txt",
    "example.req",
    "public-seed.bin",
)


class Provider:
    """Lists the hosts it was asked for until they are deleted; records every call."""

    def __init__(self, balance="50.00", price="4.39", pods=None, cost=4.39, deletes=True):
        self.balance, self.listed = Decimal(balance), Decimal(price)
        self.pods = dict(pods or {})
        self.cost, self.deletes, self.calls = cost, deletes, []

    def account(self, timeout=60):
        self.calls.append("account")
        return self.balance, dict(self.pods)

    def price(self, gpu_name, cloud):
        self.calls.append(("price", gpu_name, cloud))
        return self.listed

    def create(self, spec):
        self.calls.append(("create", spec))
        self.pods["pod1"] = spec["name"]
        return dict(
            id="pod1", costPerHr=self.cost, gpuCount=1, vcpuCount=29, memoryInGb=287, machine=None
        )

    def address(self, pod_id):
        return "198.51.100.7", 2222

    def terminate(self, pod_id, timeout=60):
        self.calls.append(("terminate", pod_id))
        if self.deletes:
            self.pods.pop(pod_id, None)
            self.balance -= Decimal("0.25")


def host(fail=None, commitment_id=ID, programs=None, verdict="YES", inputs=None):
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
            names = [*INPUTS, "weights/" + gpu.proof.KIT["gguf"]["name"]]
            rows = {name: gpu.sha(staged / name) for name in names}
            rows = inputs(rows) if inputs else rows
            (target / "public").mkdir()
            (target / "commitment-id.txt").write_text(commitment_id + "\n")
            (target / "binaries.sha256").write_text(
                "".join(f"{digest}  {name}\n" for name, digest in sums.items())
            )
            (target / "inputs.sha256").write_text(
                "".join(f"{digest}  {name}\n" for name, digest in rows.items())
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
    order = []
    monkeypatch.setattr(
        gpu.proof, "built_verifiers", lambda output: order.append("built") or (tree, {}, {})
    )
    monkeypatch.setattr(
        gpu.request_text,
        "check_text",
        lambda output: order.append("text") or dict(prompt_tokens=25),
    )
    # Waiting costs no time here: a wait moves this clock on instead.
    real, waited = time.time, [0.0]
    monkeypatch.setattr(gpu.time, "time", lambda: real() + waited[0])
    monkeypatch.setattr(gpu.time, "sleep", lambda s: waited.__setitem__(0, waited[0] + s))
    started = []

    class Guard:
        def __init__(self, name, deadline, output, key):
            started.append(("guard", name, key))
            order.append("guard")

        def poll(self):
            return None

        def terminate(self):
            started.append("stopped")

    monkeypatch.setattr(gpu, "start_guard", Guard)
    checked = dict(patched_verdict="YES", altered_copies_rejected=8)
    monkeypatch.setattr(
        gpu.proof,
        "verify",
        lambda case, cid, output, built: order.append("verify") or dict(checked),
    )
    return dict(build=build, gguf=gguf, runs=runs, guard=started, order=order)


def go(inputs, provider, remote, balance="50.00", cap="3.00", proofs=1, cloud="SECURE"):
    return gpu.run(
        "key", balance, cap, inputs["build"], inputs["gguf"], proofs, H100, cloud, provider, remote
    )


def only_run(inputs):
    (folder,) = inputs["runs"].iterdir()
    return folder


def created(provider):
    return [c[1] for c in provider.calls if isinstance(c, tuple) and c[0] == "create"]


def test_the_planned_worst_case_is_the_price_times_every_limit_and_the_deletion_time():
    most, seconds = gpu.ceiling(Decimal("4.39"))
    assert seconds == sum(gpu.LIMITS.values()) == 2400 and gpu.POD["end_seconds"] == 60
    assert most == Decimal("3.00")  # 4.39 * 2460 / 3600 = 2.9998, rounded up
    assert gpu.ceiling(Decimal("4.40"))[0] == Decimal("3.01")


@pytest.mark.parametrize("text", ["abc", "0", "-1", "nan", "inf", None])
def test_an_amount_that_is_no_positive_number_is_refused(text):
    with pytest.raises(ValueError, match="invalid"):
        gpu.dollars(text, "cost cap")


def test_the_host_carries_the_runs_name_and_the_request_to_end_itself():
    spec = gpu.pod_spec("ssh-ed25519 AAAA", "hallmark-fp-ab", H100, "SECURE", 1_800_000_000)
    after = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(1_800_000_000 + gpu.POD["grace"]))
    assert spec["terminateAfter"] == after and spec["gpuCount"] == 1 and spec["volumeInGb"] == 0
    assert spec["name"] == "hallmark-fp-ab" and "set -e" in spec["dockerArgs"]
    assert {e["key"] for e in spec["env"]} == {"PUBLIC_KEY", "NVIDIA_DRIVER_CAPABILITIES"}


def test_a_passing_run_ends_the_host_and_records_the_spend(inputs):
    provider, remote = Provider(), host()
    output = go(inputs, provider, remote)
    result = json.loads((output / "result.json").read_text())
    spend = json.loads((output / "spend.json").read_text())
    protocol = json.loads((output / "protocol.json").read_text())
    assert result["passed"] is True and result["worker_verdicts"] == ["YES"]
    assert (result["proofs_made_on_the_host"], result["proofs_checked_here"]) == (1, 1)
    assert result["spend"] == spend and spend["host_deleted"] and spend["within_cap"]
    assert spend["balance_fall_at_end_usd"] == "0.2500" and spend["cap_usd"] == "3.00"
    assert spend["price_usd_per_hour"] == "4.39"
    assert protocol["planned_worst_case_usd"] == "3.00" and protocol["owner_approved"] is False
    assert ("terminate", "pod1") in provider.calls and provider.pods == {}
    assert inputs["guard"][-1] == "stopped"
    # Text, verifiers and the second process come before the host; the local check after it.
    assert inputs["order"] == ["text", "built", "guard", "verify"]
    (spec,) = created(provider)
    assert inputs["guard"][0] == ("guard", spec["name"], "key")
    assert spec["name"].startswith("hallmark-fp-") and len(spec["name"]) == 24
    assert remote.commands[-1].endswith("pod_job.sh prove 1")


def test_no_record_names_the_host_the_balance_or_the_key(inputs):
    output = go(inputs, Provider(balance="41.37"), host(), balance="41.37")
    for name in ("result.json", "spend.json", "protocol.json", "enrollment.json"):
        text = (output / name).read_text()
        for word in ("pod1", "198.51.100.7", "2222", "41.37", '"key"', "hallmark-fp-"):
            assert word not in text, (name, word)
    assert (output / "pod-id.txt").read_text() == "pod1\n"


@pytest.mark.parametrize(
    ("provider", "cap", "balance", "why"),
    [
        (Provider(), "2.99", "50.00", "over the cap"),
        (Provider(pods={"other": "someone's"}), "3.00", "50.00", "another host"),
        (Provider(), "3.00", "2.99", "does not cover"),
        (Provider(balance="2.99"), "3.00", "50.00", "does not cover"),
        (Provider(price="5.29"), "3.00", "50.00", "over the cap"),
    ],
)
def test_no_host_is_asked_for_when_the_worst_case_does_not_fit(inputs, provider, cap, balance, why):
    with pytest.raises(ValueError, match=why):
        go(inputs, provider, host(), balance=balance, cap=cap)
    assert created(provider) == [] and "guard" not in inputs["order"]
    failure = json.loads((only_run(inputs) / "failure.json").read_text())
    assert failure["host_asked_for"] is False and failure["spend"] is None
    assert not any(isinstance(c, tuple) and c[0] == "terminate" for c in provider.calls)


def test_a_missing_key_fails_before_any_step(inputs):
    with pytest.raises(ValueError, match="RUNPOD_API_KEY is not set"):
        gpu.run(None, "50.00", "3.00", inputs["build"], inputs["gguf"], 1, H100, "SECURE")
    assert inputs["order"] == [] and not inputs["runs"].exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda i: i["gguf"].write_bytes(b"other weights"),
        lambda i: (i["build"] / "hc/cp_prove").unlink(),
        lambda i: i["gguf"].rename(i["gguf"].with_name("other.gguf")),
    ],
)
def test_no_host_is_asked_for_with_files_that_are_not_the_enrolled_ones(inputs, change):
    change(inputs)
    gguf = inputs["gguf"] if inputs["gguf"].exists() else inputs["gguf"].with_name("other.gguf")
    provider = Provider()
    with pytest.raises(ValueError):
        gpu.run("key", "50.00", "3.00", inputs["build"], gguf, 1, H100, "SECURE", provider, host())
    assert provider.calls == []


@pytest.mark.parametrize("step", ["pod_job.sh host", "pod_job.sh libraries", "pod_job.sh prove"])
def test_a_failed_step_still_ends_the_host(inputs, step):
    provider = Provider()
    with pytest.raises(ValueError, match="ended with status 9"):
        go(inputs, provider, host(fail=step))
    assert ("terminate", "pod1") in provider.calls and provider.pods == {}
    failure = json.loads((only_run(inputs) / "failure.json").read_text())
    assert failure["host_asked_for"] is True and failure["spend"]["host_deleted"] is True
    assert not (only_run(inputs) / "result.json").exists()


def test_a_host_that_costs_more_than_listed_is_ended_at_once(inputs):
    provider, remote = Provider(cost=4.40), host()
    with pytest.raises(ValueError, match="costs more"):
        go(inputs, provider, remote)
    assert ("terminate", "pod1") in provider.calls and remote.commands == []


def test_a_host_that_cannot_be_deleted_is_said_loudly_and_the_guard_stays(inputs, capsys):
    provider = Provider(deletes=False)
    with pytest.raises(ValueError, match="THE HOST MAY STILL EXIST"):
        go(inputs, provider, host())
    assert "stopped" not in inputs["guard"]
    assert "THE HOST MAY STILL EXIST AND IS BILLED" in capsys.readouterr().out
    spend = json.loads((only_run(inputs) / "spend.json").read_text())
    assert spend["host_deleted"] is False and spend["balance_fall_at_end_usd"] is None
    # The controller stops trying after the time that the planned worst case counts.
    assert 2 <= provider.calls.count(("terminate", "pod1")) <= 1 + gpu.POD["end_seconds"] // 5


def test_a_host_left_by_a_failed_request_is_ended_and_others_are_left_alone(inputs):
    class Broken(Provider):
        def create(self, spec):
            self.calls.append(("create", spec))
            self.pods["left"] = spec["name"]
            self.pods["theirs"] = "someone's"  # appeared meanwhile; not this run's
            raise ValueError("the provider could not be reached: TimeoutError")

    provider = Broken()
    with pytest.raises(ValueError, match="TimeoutError"):
        go(inputs, provider, host())
    assert ("terminate", "left") in provider.calls and ("terminate", "theirs") not in provider.calls
    assert provider.pods == {"theirs": "someone's"}
    failure = json.loads((only_run(inputs) / "failure.json").read_text())
    assert failure["host_asked_for"] is True and failure["spend"]["host_deleted"] is True


def test_a_host_that_is_listed_late_after_a_failed_request_is_still_ended(inputs):
    class Late(Provider):
        looks = 0

        def create(self, spec):
            self.calls.append(("create", spec))
            self.name = spec["name"]
            raise ValueError("the provider could not be reached: TimeoutError")

        def account(self, timeout=60):
            self.looks += 1
            if self.looks == 3:  # the first look after the failure finds nothing
                self.pods["late"] = self.name
            return super().account(timeout)

    provider = Late()
    with pytest.raises(ValueError, match="TimeoutError"):
        go(inputs, provider, host())
    assert ("terminate", "late") in provider.calls and provider.pods == {}


def test_signals_do_not_interrupt_the_deletion(inputs):
    seen = []

    class Watching(Provider):
        def terminate(self, pod_id, timeout=60):
            seen.append(signal.getsignal(signal.SIGTERM))
            super().terminate(pod_id, timeout)

    before = signal.getsignal(signal.SIGTERM)
    go(inputs, Watching(), host())
    assert seen == [signal.SIG_IGN] and signal.getsignal(signal.SIGTERM) == before


@pytest.mark.parametrize(
    ("remote", "why"),
    [
        (host(commitment_id="0" * 64), "another commitment ID"),
        (host(programs={"cp_commit": "0" * 64}), "other programs"),
        (host(verdict="NO: proof-refused"), "did not accept proof 1"),
        (host(inputs=lambda rows: {}), "names other files"),
        (host(inputs=lambda rows: rows | {"../enrollment.json": "0" * 64}), "names other files"),
        (host(inputs=lambda rows: rows | {"example.req": "0" * 64}), "another example.req"),
    ],
)
def test_what_the_host_returns_is_held_to_the_enrollment(inputs, remote, why):
    provider = Provider()
    with pytest.raises(ValueError, match=why):
        go(inputs, provider, remote)
    assert provider.pods == {} and not (only_run(inputs) / "result.json").exists()
    assert "verify" not in inputs["order"]


def test_a_sums_file_with_another_form_of_line_is_refused(tmp_path):
    path = tmp_path / "inputs.sha256"
    path.write_text(f"{'a' * 64}  example.req\n{'b' * 64}  example.req\n")
    with pytest.raises(ValueError, match="unexpected line"):
        gpu.sums(path)
    path.write_text(f"{'a' * 64} example.req\n")
    with pytest.raises(ValueError, match="unexpected line"):
        gpu.sums(path)
    path.write_text(f"{'a' * 64}  example.req\n")
    assert gpu.sums(path) == {"example.req": "a" * 64}


def test_ending_hosts_survives_every_error_and_stops_at_its_time(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(gpu.time, "time", lambda: clock[0])
    monkeypatch.setattr(gpu.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    class Flaky(Provider):
        def terminate(self, pod_id, timeout=60):
            self.calls.append(("terminate", pod_id))
            if self.calls.count(("terminate", pod_id)) < 3:
                raise TypeError("an error of no expected kind")
            self.pods.pop(pod_id)

    flaky = Flaky(pods={"pod1": "mine", "other": "theirs"})
    assert gpu.end_hosts(flaky, "mine", 60) is True
    assert flaky.calls.count(("terminate", "pod1")) == 3 and flaky.pods == {"other": "theirs"}
    clock[0] = 0.0
    never = Provider(pods={"pod1": "mine"}, deletes=False)
    assert gpu.end_hosts(never, "mine", 60) is False
    assert 50 <= clock[0] <= 60 and never.calls.count(("terminate", "pod1")) >= 10


def test_the_second_process_acts_when_the_first_is_gone_or_the_deadline_has_passed(
    tmp_path, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(gpu.time, "time", lambda: clock[0])
    monkeypatch.setattr(gpu.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    provider = Provider(pods={"pod1": "mine", "other": "theirs"})
    monkeypatch.setattr(gpu, "RunPod", lambda key: provider)
    monkeypatch.setattr(gpu.os, "getppid", lambda: 7 if clock[0] < 10 else 1)
    gpu.watchdog("mine", 1000.0, 7, tmp_path / "guard.log")
    assert 10 <= clock[0] < 20 and provider.pods == {"other": "theirs"}  # the first one was gone
    assert "ended=True" in (tmp_path / "guard.log").read_text()
    assert "RUNPOD_API_KEY" not in gpu.os.environ
    clock[0] = 0.0
    provider.pods["pod2"] = "mine"
    monkeypatch.setenv("RUNPOD_API_KEY", "key")
    monkeypatch.setattr(gpu.os, "getppid", lambda: 7)
    gpu.watchdog("mine", 100.0, 7, tmp_path / "guard.log")
    assert 100 <= clock[0] < 110 and "pod2" not in provider.pods  # the deadline had passed


def test_a_balance_that_cannot_be_read_leaves_the_bound_alone():
    class Silent(Provider):
        def account(self, timeout=60):
            raise TypeError("no account in the reply")

    assert gpu.balance_fall(Silent(), Decimal("10")) is None
    assert gpu.balance_fall(Provider(balance="9.5"), Decimal("10")) == Decimal("0.5000")


def test_a_step_gets_no_more_time_than_the_deadline_leaves(inputs, monkeypatch):
    remote = host()
    # The host starts at once; 2,395 s later five seconds are left for the copy and the step host.
    clock = iter([1000.0, 1000.0, 3395.0, 3395.0])
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
    assert ("terminate", "pod1") in provider.calls


def test_two_proofs_need_two_accepting_verdicts(inputs):
    with pytest.raises(FileNotFoundError):  # the stand-in host returns one verdict only
        go(inputs, Provider(), host(), proofs=2)


def test_the_staged_files_are_the_callers_and_the_pins_name_them(inputs):
    go(inputs, Provider(), host())
    staged = only_run(inputs) / "stage"
    pins = gpu.sums(staged / "expected.sha256")
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


def test_the_provider_s_odd_replies_are_errors_and_never_carry_the_key(monkeypatch):
    replies = iter([b'{"data": {"myself": null}}', b"not json", b'{"errors": [{"message": "no"}]}'])

    class Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return next(replies)

    seen = []
    monkeypatch.setattr(
        gpu.urllib.request, "urlopen", lambda request, timeout: seen.append(request) or Reply()
    )
    provider = gpu.RunPod("secret-key")
    for _ in range(3):
        with pytest.raises(ValueError) as error:
            provider.account()
        assert "secret-key" not in str(error.value)
    assert seen[0].get_header("Authorization") == "Bearer secret-key"


def test_ssh_uses_no_configuration_or_agent_of_the_caller(tmp_path):
    options = gpu.Remote(tmp_path, "198.51.100.7", 2222).options
    assert options[:2] == ["-F", "/dev/null"] and "ForwardAgent=no" in options
    assert "IdentitiesOnly=yes" in options and "BatchMode=yes" in options
