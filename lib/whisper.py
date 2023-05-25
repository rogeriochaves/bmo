import subprocess
from typing import Optional


def transcribe(file) -> str:
    whispercpp = subprocess.Popen(
        args=[
            "./whisper.cpp/main",
            "-m",
            "./whisper.cpp/models/ggml-tiny.en.bin",
            "-nt",
            "-f",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    whispercpp.stdin.write(file)  # type: ignore
    whispercpp.stdin.close()  # type: ignore

    output, _ = whispercpp.communicate()
    output = output.decode()

    return output


class WhisperTranscriber:
    whispercpp: Optional[subprocess.Popen[bytes]]
    first: bool

    def __init__(self) -> None:
        self.whispercpp = None

    def restart(self):
        self.stop()
        self.whispercpp = subprocess.Popen(
            args=[
                "./whisper.cpp/stream",
                "-m",
                "./whisper.cpp/models/ggml-tiny.en.bin",
                "-t",
                "8",
                "--step",
                "500",
                "--length",
                "5000",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.first = True

    def stop(self):
        if self.whispercpp is None:
            return

        self.whispercpp.stdin.close()  # type: ignore
        self.whispercpp.kill()
        self.whispercpp = None

    def transcribe_and_stop(self):
        if self.whispercpp is None:
            return

        self.whispercpp.terminate()
        output, _ = self.whispercpp.communicate()
        output_lines = output.decode().split("\n")
        output_lines = [line.split("\x1b[2K\r")[-1].strip() for line in output_lines]
        output = "\n".join([line for line in output_lines if line != ""])
        self.whispercpp.kill()

        return output
