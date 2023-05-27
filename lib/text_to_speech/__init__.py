import multiprocessing
import subprocess
from typing import Dict, Type
from typing_extensions import Protocol
from lib.delta_logging import logging
from lib.text_to_speech.elevenlabs_api import ElevenLabsAPI
from lib.text_to_speech.native_tts import NativeTTS
from lib.text_to_speech.piper_tts import PiperTTS


logger = logging.getLogger()


class TextToSpeech(Protocol):
    min_words: int

    def __init__(self, reply_out_queue: multiprocessing.Queue) -> None:
        pass

    def start(self):
        pass

    def wait_to_finish(self):
        pass

    def stop(self):
        pass

    def consume(self, word: str):
        pass


ENGINES: Dict[str, Type[TextToSpeech]] = {
    "native": NativeTTS,
    "elevenlabs": ElevenLabsAPI,
    "piper": PiperTTS,
}


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
