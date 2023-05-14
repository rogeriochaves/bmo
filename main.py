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

picovoice_access_key = os.environ["PICOVOICE_ACCESS_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

logger = logging.getLogger()

porcupine = pvporcupine.create(
    access_key=picovoice_access_key,
    keyword_paths=wakeup_keywords(),
)
frame_length = porcupine.frame_length  # 512
buffer_size = frame_length * 32 * 5  # keeps 5s of audio


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
    recorder.start()

    logger.info("Listening ... (press Ctrl+C to exit)")

    audio_buffer = bytearray()
    try:
        while True:
            pcm = recorder.read()
            result = porcupine.process(pcm)

            print(f"🔴 {red}Recording...{reset}", end="\r", flush=True)

            audio_buffer.extend(struct.pack("h" * len(pcm), *pcm))
            if len(audio_buffer) > buffer_size:
                audio_buffer = audio_buffer[
                    frame_length:
                ]  # drop early frames to keep just most recent audio

            if result >= 0:
                logger.info("Detected hotword #%s", result)
                audio_file = create_audio_file(audio_buffer)
                logger.info("Built wav file")

                recorder.stop()
                transcription = openai.Audio.transcribe("whisper-1", audio_file)
                logger.info("Transcription: %s", transcription["text"])

                reply = chatgpt.reply(transcription["text"])
                logger.info("Reply: %s", reply["content"])

                audio_stream = elevenlabs.text_to_speech(reply["content"])

                elevenlabs.play(audio_stream)
                logger.info("Playing audio done")

                recorder.start()

    except KeyboardInterrupt:
        print("Stopping ...")
    finally:
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()
