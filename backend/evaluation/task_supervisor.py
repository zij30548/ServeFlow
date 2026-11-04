"""POSIX outer deadline for the fixed, trusted evaluation worker.

Native decoding/embedding threads cannot be killed safely inside Python. This
parent runs the isolated worker in a new process group. It never restarts a
worker, changes a ledger, or promises to undo already-sent model work/charges.
There is deliberately no arbitrary-command CLI in this module.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time


STOP_FILE = "supervisor-stop.json"
TERM_GRACE_S = 3.0
KILL_GRACE_S = 2.0


class SupervisorError(RuntimeError):
    """Fixed failure codes only; never child output or exception details."""


class _ParentCancelled(BaseException):
    pass


def _open_root(root):
    path = Path(root).absolute()
    fd = None
    transferred = False
    try:
        info = path.lstat()
        if (path.resolve() != path or any(parent.is_symlink() for parent in path.parents)
                or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise SupervisorError("supervisor_unsafe_root")
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
            raise SupervisorError("supervisor_root_changed")
        try:
            os.stat(STOP_FILE, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            transferred = True
            return fd
        raise SupervisorError("supervisor_already_stopped")
    except SupervisorError:
        raise
    except (OSError, ValueError, TypeError):
        raise SupervisorError("supervisor_unsafe_root") from None
    finally:
        if fd is not None and not transferred:
            os.close(fd)


def _group_alive(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except OSError:
        # Unknown/permission-denied is not evidence that the group is gone.
        return True


def _send_group(pid, signum):
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass
    except OSError:
        # Continue the bounded reap path, but never claim an unknown child exit.
        pass


def _terminate_group(process):
    """TERM, allow at most 3s, then KILL and allow at most 2s to reap.

    The group is checked even after the leader exits: a surviving descendant
    must not be left running merely because Popen.wait() already returned.
    Signal handlers suppress repeated parent interrupts during this cleanup.
    """
    pid = process.pid
    if type(pid) is not int or pid <= 1 or pid == os.getpgrp():
        raise SupervisorError("supervisor_group_invalid")
    _send_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + TERM_GRACE_S
    reaped = False
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
            reaped = True
        except subprocess.TimeoutExpired:
            break
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, _ParentCancelled):
            continue
        except Exception:
            break
        if not _group_alive(pid):
            return process.returncode if reaped else None, reaped, True
        # The leader has exited but a descendant remains. Give that descendant
        # the remainder of the TERM grace without spinning on wait().
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                time.sleep(min(0.05, remaining))
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, _ParentCancelled):
                continue
    if _group_alive(pid):
        _send_group(pid, signal.SIGKILL)
    deadline = time.monotonic() + KILL_GRACE_S
    while time.monotonic() < deadline:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
            reaped = True
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, _ParentCancelled):
            continue
        except Exception:
            break
        if not _group_alive(pid):
            code = process.returncode if type(process.returncode) is int else None
            return code, reaped, True
        remaining = deadline - time.monotonic()
        if remaining > 0:
            try:
                time.sleep(min(0.05, remaining))
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, _ParentCancelled):
                continue
    gone = not _group_alive(pid)
    code = process.returncode if reaped and gone and type(process.returncode) is int else None
    return code, reaped, gone


def _write_stop(root_fd, *, reason, deadline_s, result, reaped, group_gone, pid):
    record = {"schema_version": "serveflow-task-supervisor-stop-v1",
              "recorded_at": datetime.now(timezone.utc).isoformat(),
              "reason": reason, "deadline_s": deadline_s, **result,
              "leader_reaped": reaped, "process_group_gone": group_gone,
              "process_group_id": pid}
    raw = (json.dumps(record, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    try:
        fd = os.open(STOP_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=root_fd)
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(root_fd)
    except (OSError, ValueError):
        # Leave any partial/existing receipt untouched. Never repair or retry.
        raise SupervisorError("supervisor_stop_persist_failed") from None


def supervise(command, *, root: Path, deadline_s=900):
    """Supervise only this process's exact Python/module worker command.

    Cwd is inherited from the trusted CLI (backend), not the private batch root.
    Catchable parent cancellation is cleaned up. SIGKILL/power loss cannot run
    Python cleanup; persistent send claims still require separate reconciliation.
    """
    expected = [[sys.executable, "-B", "-u", "-m", module, "_worker"]
                for module in ("evaluation.task_trial", "evaluation.task_recheck", "evaluation.task_repair",
                               "evaluation.retrieval_trial", "evaluation.task_business_regression",
                               "evaluation.task_observation_recheck", "evaluation.task_deadline_recheck",
                               "evaluation.quality_trial", "evaluation.quality_phase_trial",
                               "evaluation.tool_scope_recheck", "evaluation.retrieval_phase_trial",
                               "evaluation.diagnostic_repair_trial", "evaluation.order_answer_recheck")]
    if (os.name != "posix" or not hasattr(os, "killpg") or not hasattr(os, "O_NOFOLLOW")
            or not hasattr(os, "O_DIRECTORY")):
        raise SupervisorError("supervisor_posix_required")
    if threading.current_thread() is not threading.main_thread():
        raise SupervisorError("supervisor_main_thread_required")
    if (type(command) is not list or command not in expected or not os.path.isabs(sys.executable)
            or not os.path.isfile(sys.executable) or not os.access(sys.executable, os.X_OK)):
        raise SupervisorError("supervisor_fixed_worker_required")
    # Four fixed arms contain four times the old 12-case plan. Only that
    # explicit workers may opt into their separately frozen 3600s outer boundary;
    # model/network stage deadlines and every old worker's 900s stay unchanged.
    quality_workers = [[sys.executable, "-B", "-u", "-m", module, "_worker"]
                       for module in ("evaluation.quality_trial", "evaluation.quality_phase_trial",
                                      "evaluation.retrieval_phase_trial")]
    maximum = 3600 if command in quality_workers else 900
    if type(deadline_s) not in (int, float) or not math.isfinite(deadline_s) or not 0 < deadline_s <= maximum:
        raise SupervisorError("supervisor_deadline_invalid")
    root_fd = _open_root(root)
    process = None
    old_handlers = {}
    state = {"stopping": False, "spawning": True, "cancelled": False}

    def cancelled(_signum, _frame):
        state["cancelled"] = True
        if not state["stopping"] and not state["spawning"]:
            raise _ParentCancelled()

    try:
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, cancelled)
            try:
                if not state["cancelled"]:
                    process = subprocess.Popen(command, shell=False, start_new_session=True,
                                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                               close_fds=True)
            finally:
                # While Popen owns the fork/exec-to-object window, handlers only
                # record cancellation. Do not mask signals: children inherit
                # that mask across exec and would then ignore graceful TERM.
                state["spawning"] = False
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, _ParentCancelled):
            state["cancelled"] = True
        except Exception:
            raise SupervisorError("supervisor_start_failed") from None
        if process is None:
            result = {"exit_code": None, "timed_out": False, "cancelled": True}
            _write_stop(root_fd, reason="parent_cancelled", deadline_s=deadline_s,
                        result=result, reaped=True, group_gone=True, pid=None)
            return result
        timed_out = False
        unexpected = False
        if not state["cancelled"]:
            try:
                code = process.wait(timeout=float(deadline_s))
                return {"exit_code": code if type(code) is int else None,
                        "timed_out": False, "cancelled": False}
            except subprocess.TimeoutExpired:
                timed_out = True
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, _ParentCancelled):
                state["cancelled"] = True
            except Exception:
                unexpected = True
        state["stopping"] = True
        code, reaped, gone = _terminate_group(process)
        result = {"exit_code": code, "timed_out": timed_out, "cancelled": state["cancelled"]}
        reason = "supervisor_error" if unexpected else "deadline_exceeded" if timed_out else "parent_cancelled"
        _write_stop(root_fd, reason=reason, deadline_s=deadline_s, result=result,
                    reaped=reaped, group_gone=gone, pid=process.pid)
        if unexpected:
            raise SupervisorError("supervisor_wait_failed") from None
        return result
    finally:
        try:
            for sig, previous in old_handlers.items():
                signal.signal(sig, previous)
        finally:
            os.close(root_fd)
