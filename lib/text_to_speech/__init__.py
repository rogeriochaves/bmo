import multiprocessing
import subprocess
from typing_extensions import Protocol
from lib.delta_logging import logging


logger = logging.getLogger()


class TextToSpeech(Protocol):
    min_words: int

    def start(self):
        pass

    def request_to_stop(self):
        pass

    def consume(self, word: str):
        pass


def play_audio_file_non_blocking(audio_file):
    filename = f"static/{audio_file}"
    subprocess.Popen(
        ["ffplay", filename, "-autoexit", "-nodisp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def play_audio_file(filename, reply_out_queue: multiprocessing.Queue):
    filename = f"static/{filename}"
    ffplay = subprocess.Popen(
        ["ffplay", filename, "-autoexit", "-nodisp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    reply_out_queue.put(("reply_audio_started", ffplay.pid))
    ffplay.wait()
    logger.info("Playing audio done")
    reply_out_queue.put(("reply_audio_ended", None))
