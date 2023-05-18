import threading
from typing import Any, List, Literal, Optional
from lib.delta_logging import logging, red, reset  # has to be the first import
from dotenv import load_dotenv
from lib.interruption_detection import InterruptionDetection  # has to be the second

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
import nanoid

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
# TODO: reduce to 1s and stop if needed
silence_limit = 2 * sample_rate // frame_length  # 2 seconds of silence
speaking_minimum = 0.5 * sample_rate // frame_length  # 0.5 seconds of speaking
silence_time_to_standby = (
    10 * sample_rate // frame_length
)  # goes back to wakeup word checking after 10s of silence


RecordingState = Literal[
    "waiting_for_wakeup",
    "waiting_for_silence",
    "waiting_for_next_frame",
    "replying",
    "waiting_for_interruption",
]


class AudioRecording:
    state: RecordingState
    silence_frame_count: int
    speaking_frame_count: int
    recording_audio_buffer: bytearray
    recorder: PvRecorder
    reply_request_id: Optional[str] = None
    reply_thread: Optional[threading.Thread] = None
    interruption_detection: Optional[InterruptionDetection] = None

    def __init__(self, recorder: PvRecorder) -> None:
        self.recorder = recorder
        self.reset("waiting_for_silence")

    def reset(self, state):
        self.recorder.start()
        self.state = state
        self.silence_frame_count = 0
        self.speaking_frame_count = 0
        self.recording_audio_buffer = bytearray()
        self.interruption_detection = None

    def stop(self):
        self.recorder.stop()
        self.reply_request_id = None
        if self.reply_thread:
            self.reply_thread.join()
        if self.interruption_detection:
            self.interruption_detection.stop()

    def next_frame(self):
        pcm = self.recorder.read()
        rms = np.sqrt(np.mean(np.array(pcm) ** 2))
        is_silence = rms < silence_threshold

        if self.interruption_detection is not None:
            if self.interruption_detection.is_done():
                self.reply_request_id = None
                self.reset("waiting_for_silence")
                return
            else:
                interrupted = self.interruption_detection.check_for_interruption(
                    pcm, is_silence
                )
                if interrupted:
                    logger.info("Interrupted")
                    self.reply_request_id = None
                    self.reset("waiting_for_silence")
                    return
                else:
                    return

        self.recording_audio_buffer.extend(struct.pack("h" * len(pcm), *pcm))
        self.drop_early_recording_audio_frames()

        if self.state == "waiting_for_wakeup":
            self.waiting_for_wakeup(pcm)

        elif self.state == "waiting_for_silence":
            self.waiting_for_silence(is_silence)

        elif self.state == "waiting_for_next_frame":
            self.state = "replying"
            self.speaking_frame_count = 0
            self.reply_request_id = nanoid.generate()

        elif self.state == "replying":
            if self.reply_thread is not None:
                self.reply_thread.join()  # wait for the previous thread to finish
            self.recorder.stop()
            elevenlabs.play_audio_file_non_blocking("beep.mp3")
            audio_file = create_audio_file(self.recording_audio_buffer)
            logger.info("Built wav file")

            self.reply_thread = threading.Thread(target=reply, args=(self, audio_file))
            self.reply_thread.daemon = True
            self.reply_thread.start()

            self.reset("waiting_for_interruption")
            self.recorder.start()

        elif self.state == "waiting_for_interruption":
            pass

    def drop_early_recording_audio_frames(self):
        if len(self.recording_audio_buffer) > (
            buffer_size_when_not_listening
            if self.state == "waiting_for_wakeup"
            else buffer_size_on_active_listening
        ):
            self.recording_audio_buffer = self.recording_audio_buffer[
                frame_length:
            ]  # drop early frames to keep just most recent audio

    def waiting_for_wakeup(self, pcm: List[Any]):
        print(f"⚪️ Waiting for wake up word...", end="\r", flush=True)
        trigger = porcupine.process(pcm)
        if trigger >= 0:
            logger.info("Detected wakeup word #%s", trigger)
            self.state = "waiting_for_next_frame"

    def waiting_for_silence(self, is_silence: bool):
        emoji = "🔈" if is_silence else "🔊"
        print(f"🔴 {red}Listening... {emoji} {reset}", end="\r", flush=True)

        if is_silence:
            self.silence_frame_count += 1
            if (
                self.speaking_frame_count < speaking_minimum
                and self.silence_frame_count >= silence_limit * 2
            ):
                self.speaking_frame_count = 0
        else:
            if self.speaking_frame_count == 0:
                self.recording_audio_buffer = self.recording_audio_buffer[
                    -frame_length:
                ]

            self.speaking_frame_count += 1
            self.silence_frame_count = 0
            if self.speaking_frame_count >= speaking_minimum:
                self.reply_request_id = None
                self.reply_audio = None

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


def reply(audio_recording: AudioRecording, audio_file: BytesIO):
    request_id = audio_recording.reply_request_id
    transcription: Any = openai.Audio.transcribe("whisper-1", audio_file)
    logger.info("Transcription: %s", transcription["text"])
    chatgpt.add_user_message(transcription["text"])

    if request_id != audio_recording.reply_request_id:
        return

    reply = chatgpt.reply()
    if request_id != audio_recording.reply_request_id:
        return

    elevenlabs.play_audio_file_non_blocking("beep2.mp3")
    logger.info("Reply: %s", reply["content"])

    audio_stream = elevenlabs.text_to_speech(reply["content"])

    elevenlabs.play(audio_recording, audio_stream)
    logger.info("Playing audio done")


def create_audio_file(recording_audio_buffer):
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


def main():
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    audio_recording = AudioRecording(recorder)

    try:
        while True:
            audio_recording.next_frame()
    except KeyboardInterrupt:
        print("Stopping ...")
        audio_recording.stop()
    finally:
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()
