import subprocess
from typing import Dict, Type
from typing_extensions import Protocol

from lib.delta_logging import logging
from lib.speech_recognition.lightning_whisper_mlx import LightningWhisperMlx
from lib.speech_recognition.whisper_api import WhisperAPI
from lib.speech_recognition.whisper_cpp import WhisperCpp

logger = logging.getLogger()


class SpeechRecognition(Protocol):
    def restart(self):
        pass

    def stop(self):
        pass

    def consume(self, audio_buffer):
        pass

    def transcribe_and_stop(self) -> str:
        return ""

ENGINES : Dict[str, Type[SpeechRecognition]] = {
    "whisper": WhisperAPI,
    "whisper-cpp": WhisperCpp,
    "lightning-whisper-mlx": LightningWhisperMlx
}

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
