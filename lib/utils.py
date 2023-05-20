from typing import Optional
import psutil
import numpy as np


def terminate_pid_safely(pid: Optional[int]):
    if pid is None:
        return

    if not psutil.pid_exists(pid):
        return

    process = psutil.Process(pid)
    if process.status() == psutil.STATUS_RUNNING:
        process.terminate()


def calculate_volume(pcm):
    return np.sqrt(np.mean(np.array(pcm) ** 2))
