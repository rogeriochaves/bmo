from io import BytesIO
import os
from threading import Thread
import time
from typing import Dict, Union
import wave

from openai import OpenAI

from lib.delta_logging import logging

logger = logging.getLogger()
openai = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)


class WhisperAPI:
    transcription_index: int
    transcription_cut: int
    transcription_results: Dict[int, Union[Exception, str]]

    def __init__(self) -> None:
        self.transcription_index = 0

    def restart(self):
        self.stop()
        self.transcription_cut = self.transcription_index
        self.transcription_results = {}

    def stop(self):
        pass

    def consume(self, audio_buffer):
        if (
            len(audio_buffer) < 0.1 * 512 * 32
        ):  # a minimum of 0.1s is required for whisper to process
            return
        thread = Thread(
            target=self.transcribe_async, args=(audio_buffer, self.transcription_index)
        )
        thread.start()
        self.transcription_index += 1

    def transcribe_async(self, audio_buffer, index):
        try:
            audio_file = self.create_audio_file(audio_buffer)

            transcription = openai.audio.transcriptions.create(
                model="whisper-1", file=audio_file
            )

            if index >= self.transcription_cut:
                self.transcription_results[index] = transcription.text  # type: ignore
        except Exception as err:
            if index >= self.transcription_cut:
                self.transcription_results[index] = err

    def transcribe_and_stop(self):
        now = time.time()
        minimum_transcriptions = max(
            self.transcription_index - self.transcription_cut, 1
        )

        while (
            self.transcription_index >= minimum_transcriptions
            and len(self.transcription_results.values()) < minimum_transcriptions
        ):
            if time.time() - now > 3:
                break
            time.sleep(0.03)

        results = list(self.transcription_results.values())
        if len(results) == 1 and isinstance(results[0], Exception):
            raise results[0]

        result = " ".join([r for r in results if type(r) == str])
        logger.info("Transcription: %s", result)
        return result

    def create_audio_file(self, recording_audio_buffer):
        virtual_file = BytesIO()
        wav_file = wave.open(virtual_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(recording_audio_buffer)
        wav_file.close()
        virtual_file.name = "recording.wav"
        virtual_file.seek(0)

        return virtual_file
