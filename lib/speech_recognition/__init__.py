from abc import abstractmethod
import subprocess
from typing_extensions import Protocol

from lib.delta_logging import logging

logger = logging.getLogger()


class SpeechRecognition(Protocol):
    def restart(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def consume(self, audio_buffer):
        pass

    def transcribe_and_stop(self) -> str:
        return ""


def transcribe(file) -> str:
    whispercpp = subprocess.Popen(
        args=[
            "./whisper.cpp/main",
            "-m",
            "./whisper.cpp/models/ggml-medium.bin",
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
