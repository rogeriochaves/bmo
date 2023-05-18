import psutil


def terminate_pid_safely(pid: int):
    if psutil.pid_exists(pid):
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_RUNNING:
            process.terminate()
