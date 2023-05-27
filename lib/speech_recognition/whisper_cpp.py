import subprocess
from typing import Optional
from lib.delta_logging import logging

logger = logging.getLogger()


class WhisperCpp:
    whispercpp: Optional[subprocess.Popen]
    first: bool

    def __init__(self) -> None:
        self.whispercpp = None

    def restart(self):
        self.stop()
        self.whispercpp = subprocess.Popen(
            args=[
                "./whisper.cpp/stream",
                "-m",
                "./whisper.cpp/models/ggml-medium.en.bin",
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

    def consume(self, audio_buffer):
        pass

    def transcribe_and_stop(self):
        if self.whispercpp is None:
            return ""

        self.whispercpp.terminate()
        output, _ = self.whispercpp.communicate()
        output_lines = output.decode().split("\n")
        output_lines = [line.split("\x1b[2K\r")[-1].strip() for line in output_lines]
        output = "\n".join([line for line in output_lines if line != ""])
        logger.info("Transcription: %s", output)
        self.whispercpp.kill()

        return output
