#!/usr/bin/env python3
"""Supervise one MATLAB-owned Python baseline adapter process.

v73l hardens the Windows MATLAB-liveness check used for long external
baselines.  A transient/indeterminate Win32 handle query must never cancel a
healthy adapter.  The supervisor therefore:
  * declares all Win32 ctypes signatures explicitly (64-bit HANDLE safe),
  * treats an indeterminate WAIT_FAILED result as "unknown/alive",
  * requires repeated parent/control failures before cancellation,
  * prints the exact cancellation reason,
  * preserves adapter_pid.txt for post-mortem diagnostics,
  * ignores spurious Windows console signals while MATLAB remains alive.
"""
from __future__ import annotations
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

_STOP = False
_SIGNAL_EVENTS = []


def _pid_state(pid: int) -> Tuple[bool, str]:
    """Return (alive, diagnostic).

    On Windows, uncertain API results are deliberately interpreted as alive.
    The control file remains the primary MATLAB-owned cancellation signal.
    """
    if pid <= 0:
        return True, "pid_not_required"
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_SYNCHRONIZE = 0x00100000
            WAIT_OBJECT_0 = 0x00000000
            WAIT_TIMEOUT = 0x00000102
            WAIT_FAILED = 0xFFFFFFFF

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            k32.OpenProcess.restype = wintypes.HANDLE
            k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            k32.WaitForSingleObject.restype = wintypes.DWORD
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL

            h = k32.OpenProcess(PROCESS_SYNCHRONIZE, False, int(pid))
            if not h:
                err = int(ctypes.get_last_error())
                # ERROR_INVALID_PARAMETER (87) commonly means the PID no longer
                # exists. Access-denied/other errors are indeterminate and must
                # not terminate a healthy long-running adapter.
                if err == 87:
                    return False, "OpenProcess_invalid_pid"
                return True, "OpenProcess_indeterminate_error_%d" % err
            try:
                result = int(k32.WaitForSingleObject(h, 0))
            finally:
                k32.CloseHandle(h)
            if result == WAIT_TIMEOUT:
                return True, "running"
            if result == WAIT_OBJECT_0:
                return False, "process_signaled"
            if result == WAIT_FAILED:
                return True, "WaitForSingleObject_failed_unknown"
            return True, "unexpected_wait_result_%d" % result
        except Exception as exc:
            return True, "windows_pid_check_exception_%s" % type(exc).__name__
    try:
        os.kill(pid, 0)
        return True, "running"
    except OSError:
        return False, "kill0_failed"


def _kill_tree(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    pid = int(proc.pid)
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


def _handle(sig, _frame):
    """Handle external console signals without aborting healthy Windows runs.

    MATLAB owns cancellation through the unique control file and the guarded
    Java-process cleanup.  On Windows, CTRL_C/CTRL_BREAK propagation can reach
    sibling console processes even when the MATLAB section was not stopped.
    Therefore a Windows console SIGINT/SIGTERM is diagnostic only while the
    MATLAB-owned control/liveness checks remain healthy.  POSIX behavior stays
    unchanged.
    """
    global _STOP
    name = getattr(sig, "name", None) or str(sig)
    _SIGNAL_EVENTS.append((time.time(), int(sig), name))
    if os.name == "nt":
        print(
            "[Python baseline supervisor] ignored Windows console signal %s (%d); "
            "MATLAB control-file/PID checks remain authoritative" % (name, int(sig)),
            flush=True,
        )
        return
    _STOP = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--parent-pid", type=int, required=True)
    ap.add_argument("--control-file", required=True)
    ap.add_argument("--adapter-pid-file", required=True)
    a = ap.parse_args()

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None), getattr(signal, "SIGBREAK", None)):
        if sig is not None:
            try:
                signal.signal(sig, _handle)
            except Exception:
                pass

    control = Path(a.control_file)
    pid_file = Path(a.adapter_pid_file)
    cmd = [
        a.python,
        a.adapter,
        "--config",
        a.config,
        "--parent-pid",
        str(a.parent_pid),
        "--control-file",
        a.control_file,
    ]
    kw = dict(stdout=sys.stdout, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True

    proc = None
    cancel_reason = ""
    # Five seconds of repeated failure prevents one transient filesystem/PID
    # query from cancelling a multi-hour run.
    grace_checks = 25
    missing_control_checks = 0
    dead_parent_checks = 0
    try:
        proc = subprocess.Popen(cmd, **kw)
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(
            "[Python baseline supervisor] supervisor_pid=%d; adapter_pid=%d; matlab_pid=%d"
            % (os.getpid(), proc.pid, a.parent_pid),
            flush=True,
        )
        while proc.poll() is None:
            if _STOP:
                cancel_reason = "supervisor received an honored POSIX termination signal"
                break

            if control.exists():
                missing_control_checks = 0
            else:
                missing_control_checks += 1

            parent_alive, parent_diag = _pid_state(a.parent_pid)
            if parent_alive:
                dead_parent_checks = 0
            else:
                dead_parent_checks += 1

            if missing_control_checks >= grace_checks:
                cancel_reason = "MATLAB control file missing for %.1f s: %s" % (
                    0.2 * missing_control_checks,
                    control,
                )
                break
            if dead_parent_checks >= grace_checks:
                cancel_reason = "MATLAB parent PID %d reported dead for %.1f s (%s)" % (
                    a.parent_pid,
                    0.2 * dead_parent_checks,
                    parent_diag,
                )
                break
            time.sleep(0.2)

        if cancel_reason:
            print("[Python baseline supervisor] cancellation reason: %s" % cancel_reason, flush=True)
            _kill_tree(proc)
            return 130
        return int(proc.returncode or 0)
    finally:
        if proc is not None and proc.poll() is None:
            _kill_tree(proc)
        # Keep adapter_pid.txt in the unique work directory so MATLAB and the
        # user can diagnose completed/cancelled runs. It is never reused.


if __name__ == "__main__":
    raise SystemExit(main())
