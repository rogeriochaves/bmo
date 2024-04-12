import tempfile
import wave
from typing import Optional

from lightning_whisper_mlx import LightningWhisperMLX

from lib.delta_logging import logging

logger = logging.getLogger()


class LightningWhisperMlx:
    whisper: Optional[LightningWhisperMLX]
    audio_buffer = bytearray()

    def __init__(self) -> None:
        self.whisper = None

    def restart(self):
        self.stop()
        self.whisper = LightningWhisperMLX(model="base", batch_size=12, quant=None)
        self.audio_buffer = bytearray()

    def stop(self):
        if self.whisper is None:
            return

        self.whisper = None

    def consume(self, audio_buffer):
        self.audio_buffer.extend(audio_buffer)

    def transcribe_and_stop(self):
        if self.whisper is None:
            return

        temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".wav")

        with wave.open(temp_file.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(self.audio_buffer)

        result = str(self.whisper.transcribe(temp_file.name)["text"])

        logger.info("Transcription: %s", result)
        return result
