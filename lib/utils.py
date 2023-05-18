from typing import Optional
import psutil


def terminate_pid_safely(pid: Optional[int]):
    if pid is None:
        return

    if not psutil.pid_exists(pid):
        return

    process = psutil.Process(pid)
    if process.status() == psutil.STATUS_RUNNING:
        process.terminate()
