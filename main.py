from typing import Any, Literal
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


ListeningMode = Literal["waiting_for_wakeup", "reply_on_silence"]


def conversation_loop(recorder: PvRecorder):
    silence_frame_count = 0
    speaking_frame_count = 0
    listening_mode: ListeningMode = "reply_on_silence"

    logger.info("Listening ... (press Ctrl+C to exit)")

    audio_buffer = bytearray()

    while True:
        pcm = recorder.read()
        print(f"🔴 {red}Recording...{reset}", end="\r", flush=True)

        trigger = -1
        if listening_mode == "waiting_for_wakeup":
            trigger = porcupine.process(pcm)
            if trigger >= 0:
                logger.info("Detected wakeup word #%s", trigger)
        elif listening_mode == "reply_on_silence":
            rms = np.sqrt(np.mean(np.array(pcm) ** 2))
            if rms < silence_threshold:
                silence_frame_count += 1
            else:
                if speaking_frame_count >= speaking_minimum:
                    silence_frame_count = 0
                speaking_frame_count += 1

            if (
                silence_frame_count >= silence_limit
                and speaking_frame_count >= speaking_minimum
            ):
                logger.info("Detected silence a while after speaking, giving a reply")
                trigger = 0

            if silence_frame_count >= silence_time_to_standby:
                logger.info(
                    "Long silence time, going back to waiting for the wakeup word"
                )
                silence_frame_count = 0
                speaking_frame_count = 0
                listening_mode = "waiting_for_wakeup"

        audio_buffer.extend(struct.pack("h" * len(pcm), *pcm))
        if len(audio_buffer) > (
            buffer_size_on_active_listening
            if listening_mode == "reply_on_silence"
            else buffer_size_when_not_listening
        ):
            audio_buffer = audio_buffer[
                frame_length:
            ]  # drop early frames to keep just most recent audio

        if trigger >= 0:
            elevenlabs.play_audio_file_non_blocking("beep.mp3")
            audio_file = create_audio_file(audio_buffer)
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

            listening_mode = "reply_on_silence"
            silence_frame_count = 0
            speaking_frame_count = 0
            audio_buffer = bytearray()

            recorder.start()


def main():
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
    recorder.start()
    try:
        conversation_loop(recorder)
    except KeyboardInterrupt:
        print("Stopping ...")
    finally:
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()
