#!/usr/bin/env python3
"""Supervise one MATLAB-owned PySR adapter process.

The supervisor never executes PySR itself. It remains responsive while the
adapter is blocked in model.fit(). If MATLAB stops, closes, crashes, or removes
the per-run control file, the supervisor terminates the exact adapter process
tree and only then exits.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

_STOP_REQUESTED = False


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return True
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = wintypes.HANDLE
            handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _quote_ps_literal(value: str) -> str:
    return value.replace("'", "''")


def _kill_exact_config_processes(config_path: str) -> None:
    """Fallback: kill only python processes whose command line has this config."""
    if not config_path or os.name != "nt":
        return
    token = _quote_ps_literal(os.path.abspath(config_path))
    ps = (
        f"$t='{token}'; "
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(w)?\\.exe$' -and "
        "$_.CommandLine -like ('*'+$t+'*') } | "
        "ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null }"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
    except Exception:
        pass


def _terminate_tree(proc: Optional[subprocess.Popen], config_path: str) -> None:
    if proc is None:
        _kill_exact_config_processes(config_path)
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
        time.sleep(0.4)
        if _pid_alive(pid):
            try:
                proc.kill()
            except Exception:
                pass
        _kill_exact_config_processes(config_path)
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


def _signal_handler(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("--control-file", required=True)
    parser.add_argument("--adapter-pid-file", required=True)
    args = parser.parse_args()

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, _signal_handler)
            except Exception:
                pass

    control = Path(args.control_file)
    pid_file = Path(args.adapter_pid_file)
    cmd = [
        args.python,
        args.adapter,
        "--config", args.config,
        "--parent-pid", str(args.parent_pid),
        "--control-file", args.control_file,
    ]
    popen_kwargs = {
        "stdout": sys.stdout,
        "stderr": subprocess.STDOUT,
        "stdin": subprocess.DEVNULL,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc: Optional[subprocess.Popen] = None
    reason = "adapter completed"
    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
        pid_file.write_text(str(proc.pid), encoding="utf-8")
        print(
            f"[PySR supervisor] supervisor_pid={os.getpid()}; "
            f"adapter_pid={proc.pid}; matlab_pid={args.parent_pid}",
            flush=True,
        )
        while proc.poll() is None:
            if _STOP_REQUESTED:
                reason = "supervisor received termination signal"
                break
            if not control.exists():
                reason = "MATLAB control file removed"
                break
            if not _pid_alive(args.parent_pid):
                reason = "parent MATLAB process disappeared"
                break
            time.sleep(0.20)

        if proc.poll() is None:
            print(f"[PySR supervisor] {reason}; terminating adapter tree.", flush=True)
            _terminate_tree(proc, args.config)
            try:
                proc.wait(timeout=5)
            except Exception:
                _terminate_tree(proc, args.config)
            return 130
        return int(proc.returncode or 0)
    finally:
        if proc is not None and proc.poll() is None:
            _terminate_tree(proc, args.config)
        try:
            if pid_file.exists():
                pid_file.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
