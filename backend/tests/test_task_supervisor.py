"""Fake-process supervisor checks; never launch the evaluation worker."""
import asyncio
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys

import pytest

from evaluation import task_supervisor as supervisor


COMMAND = [sys.executable, "-B", "-u", "-m", "evaluation.task_trial", "_worker"]


class Clock:
    def __init__(self): self.now = 0.0
    def monotonic(self): return self.now
    def sleep(self, seconds): self.now += seconds


class FakeProcess:
    pid = 54321
    def __init__(self, clock, actions):
        self.clock, self.actions = clock, list(actions)
        self.returncode = None
        self.group_alive = True
        self.waits = []
    def wait(self, timeout):
        self.waits.append(timeout)
        if self.actions:
            action = self.actions.pop(0)
            if action == "timeout":
                self.clock.now += timeout
                raise subprocess.TimeoutExpired(COMMAND, timeout)
            if isinstance(action, BaseException):
                self.clock.now += min(timeout, .1)
                raise action
            if callable(action):
                return action(self, timeout)
            self.returncode = action
        return self.returncode


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("supervisor tests must not contact services")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    root = tmp_path / "batch"; root.mkdir(mode=0o700)
    (root / "ledger.json").write_bytes(b"keep original ledger")
    clock = Clock()
    handlers = {signal.SIGINT: "old-int", signal.SIGTERM: "old-term"}
    signal_changes, sends, starts = [], [], []
    state = {"process": FakeProcess(clock, [0]), "signal_during_spawn": False}
    def popen(command, **kwargs):
        starts.append((list(command), kwargs))
        if state["signal_during_spawn"]:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return state["process"]
    def change_handler(sig, function):
        old = handlers.get(sig)
        handlers[sig] = function
        signal_changes.append((sig, function))
        return old
    def killpg(pid, sig):
        assert pid == state["process"].pid
        if sig == 0:
            if not state["process"].group_alive: raise ProcessLookupError()
            return
        sends.append((pid, sig))
        if sig == signal.SIGKILL:
            state["process"].group_alive = False
            if state["process"].returncode is None: state["process"].returncode = -9
    monkeypatch.setattr(supervisor.subprocess, "Popen", popen)
    monkeypatch.setattr(supervisor.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(supervisor.time, "sleep", clock.sleep)
    monkeypatch.setattr(supervisor.os, "killpg", killpg)
    monkeypatch.setattr(supervisor.os, "getpgrp", lambda: 12345)
    monkeypatch.setattr(supervisor.signal, "getsignal", lambda sig: handlers[sig])
    monkeypatch.setattr(supervisor.signal, "signal", change_handler)
    def mask_forbidden(*args):
        raise AssertionError("do not make the child inherit blocked cancellation signals")
    monkeypatch.setattr(supervisor.signal, "pthread_sigmask", mask_forbidden)
    return root, clock, state, handlers, sends, starts, signal_changes


def assert_ledger(root):
    assert (root / "ledger.json").read_bytes() == b"keep original ledger"


def test_normal_exit_uses_only_fixed_worker_group_no_logs_and_no_stop_receipt(fake):
    root, _, _, handlers, sends, starts, _ = fake
    assert supervisor.supervise(COMMAND, root=root) == {"exit_code": 0, "timed_out": False, "cancelled": False}
    assert len(starts) == 1
    command, options = starts[0]
    assert command == COMMAND and options == {"shell": False, "start_new_session": True,
        "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "close_fds": True}
    assert not sends and not (root / supervisor.STOP_FILE).exists()
    assert handlers == {signal.SIGINT: "old-int", signal.SIGTERM: "old-term"}
    assert_ledger(root)


def test_ordinary_nonzero_worker_exit_is_not_restarted_or_rewritten(fake):
    root, clock, state, _, sends, starts, _ = fake
    state["process"] = FakeProcess(clock, [2])
    assert supervisor.supervise(COMMAND, root=root)["exit_code"] == 2
    assert len(starts) == 1 and not sends and not (root / supervisor.STOP_FILE).exists()
    assert_ledger(root)


def test_timeout_terms_then_kills_group_and_reaps_with_private_stop(fake):
    root, clock, state, handlers, sends, starts, _ = fake
    state["process"] = FakeProcess(clock, ["timeout", "timeout", -9])
    outcome = supervisor.supervise(COMMAND, root=root)
    assert outcome == {"exit_code": -9, "timed_out": True, "cancelled": False}
    assert state["process"].waits == [900, 3, 2]
    assert sends == [(54321, signal.SIGTERM), (54321, signal.SIGKILL)]
    assert len(starts) == 1 and clock.now == 903
    path = root / supervisor.STOP_FILE
    record = json.loads(path.read_text())
    assert record["reason"] == "deadline_exceeded" and record["leader_reaped"] and record["process_group_gone"]
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert handlers == {signal.SIGINT: "old-int", signal.SIGTERM: "old-term"}
    assert_ledger(root)


def test_exited_leader_does_not_abandon_term_ignoring_descendant(fake):
    root, clock, state, _, sends, _, _ = fake
    # The leader exits on TERM, but its group remains alive until KILL.
    state["process"] = FakeProcess(clock, ["timeout", -15])
    outcome = supervisor.supervise(COMMAND, root=root, deadline_s=.1)
    assert outcome == {"exit_code": -15, "timed_out": True, "cancelled": False}
    assert sends == [(54321, signal.SIGTERM), (54321, signal.SIGKILL)]
    assert clock.now == pytest.approx(3.1)
    assert_ledger(root)


@pytest.mark.parametrize("cancel", [KeyboardInterrupt(), asyncio.CancelledError(), SystemExit()])
def test_parent_cancellation_cleans_group_and_persists_without_retry(fake, cancel):
    root, clock, state, _, sends, starts, _ = fake
    state["process"] = FakeProcess(clock, [cancel, "timeout", -9])
    result = supervisor.supervise(COMMAND, root=root)
    assert result == {"exit_code": -9, "timed_out": False, "cancelled": True}
    assert len(starts) == 1 and [sig for _, sig in sends] == [signal.SIGTERM, signal.SIGKILL]
    assert json.loads((root / supervisor.STOP_FILE).read_text())["reason"] == "parent_cancelled"
    assert_ledger(root)


def test_signal_during_spawn_is_deferred_until_child_object_is_assigned(fake):
    root, clock, state, _, sends, starts, _ = fake
    state["process"] = FakeProcess(clock, ["timeout", -9])
    state["signal_during_spawn"] = True
    result = supervisor.supervise(COMMAND, root=root)
    assert result["cancelled"] and not result["timed_out"]
    assert len(starts) == 1 and [sig for _, sig in sends] == [signal.SIGTERM, signal.SIGKILL]
    assert state["process"].waits == [3, 2]


def test_repeated_parent_signal_during_cleanup_does_not_skip_kill_or_reap(fake):
    root, clock, state, handlers, sends, _, _ = fake
    def repeated_signal(process, timeout):
        handlers[signal.SIGINT](signal.SIGINT, None)
        process.clock.now += timeout
        raise subprocess.TimeoutExpired(COMMAND, timeout)
    state["process"] = FakeProcess(clock, [KeyboardInterrupt(), repeated_signal, -9])
    result = supervisor.supervise(COMMAND, root=root)
    assert result["cancelled"] and [sig for _, sig in sends] == [signal.SIGTERM, signal.SIGKILL]
    assert state["process"].returncode == -9


def test_term_exit_reaps_without_unnecessary_kill(fake):
    root, clock, state, _, sends, _, _ = fake
    def graceful_exit(process, timeout):
        process.returncode = -15
        process.group_alive = False
        return -15
    state["process"] = FakeProcess(clock, ["timeout", graceful_exit])
    assert supervisor.supervise(COMMAND, root=root, deadline_s=.1) == {
        "exit_code": -15, "timed_out": True, "cancelled": False}
    assert sends == [(54321, signal.SIGTERM)]
    assert json.loads((root / supervisor.STOP_FILE).read_text())["leader_reaped"]


def test_unconfirmed_kill_is_recorded_as_unknown_not_claimed_reaped(fake, monkeypatch):
    root, clock, state, _, _, _, _ = fake
    state["process"] = FakeProcess(clock, ["timeout", "timeout", "timeout"])
    signals = []
    def unconfirmed_group(pid, sig):
        if sig: signals.append(sig)
    monkeypatch.setattr(supervisor.os, "killpg", unconfirmed_group)
    assert supervisor.supervise(COMMAND, root=root, deadline_s=.1) == {
        "exit_code": None, "timed_out": True, "cancelled": False}
    record = json.loads((root / supervisor.STOP_FILE).read_text())
    assert not record["leader_reaped"] and not record["process_group_gone"]
    assert signals == [signal.SIGTERM, signal.SIGKILL] and clock.now == pytest.approx(5.1)
    assert_ledger(root)


def test_stop_receipt_race_is_never_overwritten_after_child_cleanup(fake):
    root, clock, state, _, sends, _, _ = fake
    def marker_created_then_timeout(process, timeout):
        (root / supervisor.STOP_FILE).write_bytes(b"other stop wins")
        process.clock.now += timeout
        raise subprocess.TimeoutExpired(COMMAND, timeout)
    state["process"] = FakeProcess(clock, [marker_created_then_timeout, "timeout", -9])
    with pytest.raises(supervisor.SupervisorError, match="supervisor_stop_persist_failed"):
        supervisor.supervise(COMMAND, root=root)
    assert (root / supervisor.STOP_FILE).read_bytes() == b"other stop wins"
    assert [sig for _, sig in sends] == [signal.SIGTERM, signal.SIGKILL]
    assert_ledger(root)


def test_start_failure_is_sanitized_and_handlers_restored(fake, monkeypatch):
    root, _, _, handlers, sends, _, _ = fake
    def failed_start(*args, **kwargs):
        raise OSError("private executable location")
    monkeypatch.setattr(supervisor.subprocess, "Popen", failed_start)
    with pytest.raises(supervisor.SupervisorError, match="supervisor_start_failed") as caught:
        supervisor.supervise(COMMAND, root=root)
    assert "private" not in str(caught.value) and not sends
    assert handlers == {signal.SIGINT: "old-int", signal.SIGTERM: "old-term"}
    assert not (root / supervisor.STOP_FILE).exists()
    assert_ledger(root)


@pytest.mark.parametrize("deadline", [True, 0, -1, 901, float("nan"), float("inf"), "900"])
def test_invalid_deadline_stops_before_start(fake, deadline):
    root, _, _, _, _, starts, _ = fake
    with pytest.raises(supervisor.SupervisorError, match="supervisor_deadline_invalid"):
        supervisor.supervise(COMMAND, root=root, deadline_s=deadline)
    assert not starts


@pytest.mark.parametrize("command", [
    ["/bin/sh", "-c", "echo unsafe"], COMMAND + ["--url", "https://arbitrary.example"],
    COMMAND[:-1] + ["run"], COMMAND[:4] + ["other.module", "_worker"], tuple(COMMAND),
])
def test_no_arbitrary_command_module_or_url_entrypoint(fake, command):
    root, _, _, _, _, starts, _ = fake
    with pytest.raises(supervisor.SupervisorError, match="supervisor_fixed_worker_required"):
        supervisor.supervise(command, root=root)
    assert not starts


def test_existing_stop_is_never_overwritten_and_missing_or_unsafe_root_rejected(fake, tmp_path):
    root, _, _, _, _, starts, _ = fake
    stop = root / supervisor.STOP_FILE; stop.write_bytes(b"old stop")
    with pytest.raises(supervisor.SupervisorError, match="supervisor_already_stopped"):
        supervisor.supervise(COMMAND, root=root)
    assert stop.read_bytes() == b"old stop" and not starts
    with pytest.raises(supervisor.SupervisorError, match="supervisor_unsafe_root"):
        supervisor.supervise(COMMAND, root=tmp_path / "absent")
    unsafe = tmp_path / "unsafe"; unsafe.mkdir(mode=0o755)
    with pytest.raises(supervisor.SupervisorError, match="supervisor_unsafe_root"):
        supervisor.supervise(COMMAND, root=unsafe)
    link = tmp_path / "link"; link.symlink_to(root, target_is_directory=True)
    with pytest.raises(supervisor.SupervisorError, match="supervisor_unsafe_root"):
        supervisor.supervise(COMMAND, root=link)
    assert not starts


def test_spawn_or_wait_error_never_exposes_exception_body_and_does_not_retry(fake, monkeypatch):
    root, clock, state, _, sends, starts, _ = fake
    state["process"] = FakeProcess(clock, [RuntimeError("secret driver details"), "timeout", -9])
    with pytest.raises(supervisor.SupervisorError, match="supervisor_wait_failed") as caught:
        supervisor.supervise(COMMAND, root=root)
    assert "secret" not in str(caught.value) and len(starts) == 1
    assert len(sends) == 2 and (root / supervisor.STOP_FILE).is_file()
    assert_ledger(root)


def test_unsupported_platform_is_explicit(fake, monkeypatch):
    root, _, _, _, _, starts, _ = fake
    monkeypatch.setattr(supervisor.os, "name", "non-posix")
    with pytest.raises(supervisor.SupervisorError, match="supervisor_posix_required"):
        supervisor.supervise(COMMAND, root=root)
    assert not starts


def test_routing_fix_worker_is_an_explicit_second_allowed_command(fake):
    root, _, _, _, sends, starts, _ = fake
    command = [sys.executable, "-B", "-u", "-m", "evaluation.task_recheck", "_worker"]
    assert supervisor.supervise(command, root=root) == {"exit_code": 0, "timed_out": False, "cancelled": False}
    assert starts[0][0] == command and len(starts) == 1 and not sends
    assert_ledger(root)


@pytest.mark.parametrize("module", ["evaluation.task_recheck_extra", "evaluation.intent_recheck", "evaluation.task_trial; echo bad"])
def test_second_worker_does_not_open_arbitrary_modules(fake, module):
    root, _, _, _, _, starts, _ = fake
    command = [sys.executable, "-B", "-u", "-m", module, "_worker"]
    with pytest.raises(supervisor.SupervisorError, match="supervisor_fixed_worker_required"):
        supervisor.supervise(command, root=root)
    assert starts == []
