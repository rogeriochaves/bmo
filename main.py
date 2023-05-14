from typing import Any, List, Literal
from lib.delta_logging import logging, red, reset  # has to be the first import
from dotenv import load_dotenv  # has to be the second

load_dotenv()
from lib.porcupine import wakeup_keywords
import lib.chatgpt as chatgpt
import lib.elevenlabs as elevenlabs
import os
import struct
import wave
import pvporcupine
from pvrecorder import PvRecorder
from io import BytesIO
import openai
import numpy as np

picovoice_access_key = os.environ["PICOVOICE_ACCESS_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

logger = logging.getLogger()

porcupine = pvporcupine.create(
    access_key=picovoice_access_key,
    keyword_paths=wakeup_keywords(),
)
frame_length = porcupine.frame_length  # 512
buffer_size_when_not_listening = frame_length * 32 * 5  # keeps 5s of audio
buffer_size_on_active_listening = frame_length * 32 * 60  # keeps 60s of audio
sample_rate = 16000  # sample rate for Porcupine is fixed at 16kHz
silence_threshold = 300  # maybe need to be adjusted
silence_limit = 2 * sample_rate // frame_length  # 2 seconds of silence
speaking_minimum = 0.5 * sample_rate // frame_length  # 0.5 seconds of speaking
silence_time_to_standby = (
    10 * sample_rate // frame_length
)  # goes back to wakeup word checking after 10s of silence


RecordingState = Literal[
    "waiting_for_wakeup", "waiting_for_silence", "waiting_for_next_frame", "replying"
]


class AudioRecording:
    state: RecordingState
    silence_frame_count: int
    speaking_frame_count: int
    audio_buffer: bytearray
    recorder: PvRecorder

    def __init__(self, recorder: PvRecorder) -> None:
        self.recorder = recorder
        self.reset("waiting_for_wakeup")

    def reset(self, state):
        self.recorder.start()
        self.state = state
        self.silence_frame_count = 0
        self.speaking_frame_count = 0
        self.audio_buffer = bytearray()

    def next_frame(self):
        pcm = self.recorder.read()

        self.audio_buffer.extend(struct.pack("h" * len(pcm), *pcm))
        self.drop_early_audio_frames()

        if self.state == "waiting_for_wakeup":
            self.waiting_for_wakeup(pcm)

        elif self.state == "waiting_for_silence":
            self.waiting_for_silence(pcm)

        elif self.state == "waiting_for_next_frame":
            self.state = "replying"

    def drop_early_audio_frames(self):
        if len(self.audio_buffer) > (
            buffer_size_when_not_listening
            if self.state == "waiting_for_wakeup"
            else buffer_size_on_active_listening
        ):
            self.audio_buffer = self.audio_buffer[
                frame_length:
            ]  # drop early frames to keep just most recent audio

    def waiting_for_wakeup(self, pcm: List[Any]):
        print(f"⚪️ Waiting for wake up word...", end="\r", flush=True)
        trigger = porcupine.process(pcm)
        if trigger >= 0:
            logger.info("Detected wakeup word #%s", trigger)
            self.state = "waiting_for_next_frame"

    def waiting_for_silence(self, pcm: List[Any]):
        print(f"🔴 {red}Listening...{reset}", end="\r", flush=True)

        rms = np.sqrt(np.mean(np.array(pcm) ** 2))
        if rms < silence_threshold:
            self.silence_frame_count += 1
        else:
            if self.speaking_frame_count >= speaking_minimum:
                self.silence_frame_count = 0
            self.speaking_frame_count += 1

        if (
            self.silence_frame_count >= silence_limit
            and self.speaking_frame_count >= speaking_minimum
        ):
            logger.info("Detected silence a while after speaking, giving a reply")
            self.state = "waiting_for_next_frame"

        if self.silence_frame_count >= silence_time_to_standby:
            logger.info("Long silence time, going back to waiting for the wakeup word")
            self.silence_frame_count = 0
            self.speaking_frame_count = 0
            self.state = "waiting_for_wakeup"


def conversation_loop(recorder: PvRecorder):
    audio_recording = AudioRecording(recorder)

    logger.info("Listening ... (press Ctrl+C to exit)")

    while True:
        audio_recording.next_frame()

        if audio_recording.state == "replying":
            elevenlabs.play_audio_file_non_blocking("beep.mp3")
            audio_file = create_audio_file(audio_recording.audio_buffer)
            logger.info("Built wav file")

            recorder.stop()
            transcription: Any = openai.Audio.transcribe("whisper-1", audio_file)
            logger.info("Transcription: %s", transcription["text"])

            reply = chatgpt.reply(transcription["text"])
            elevenlabs.play_audio_file_non_blocking("beep2.mp3")
            logger.info("Reply: %s", reply["content"])

            audio_stream = elevenlabs.text_to_speech(reply["content"])

            elevenlabs.play(audio_stream)
            logger.info("Playing audio done")

            audio_recording.reset("waiting_for_silence")


def create_audio_file(audio_buffer):
    virtual_file = BytesIO()
    wav_file = wave.open(virtual_file, "wb")
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(16000)
    wav_file.writeframes(audio_buffer)
    wav_file.close()
    audio_buffer = bytearray()
    virtual_file.name = "recording.wav"
    virtual_file.seek(0)

    return virtual_file


def main():
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    try:
        conversation_loop(recorder)
    except KeyboardInterrupt:
        print("Stopping ...")
    finally:
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()
