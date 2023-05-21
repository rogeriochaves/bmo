from multiprocessing import Queue
import os
import subprocess
from elevenlabs import generate, Voice, VoiceSettings
from lib.delta_logging import logging

eleven_labs_api_key = os.environ["ELEVEN_LABS_API_KEY"]

VOICE_SETTINGS_STABILITY = 0.75
VOICE_SETTINGS_SIMILARITY_BOOST = 0.75
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # pNInz6obpgDQGcFmaJgB

logger = logging.getLogger()


def text_to_speech(text: str):
    if os.getenv("DEBUG_MODE"):
        return []
    audio_stream = generate(
        api_key=eleven_labs_api_key,
        model="eleven_multilingual_v1",
        voice=Voice(
            voice_id=VOICE_ID,
            settings=VoiceSettings(
                stability=VOICE_SETTINGS_STABILITY,
                similarity_boost=VOICE_SETTINGS_SIMILARITY_BOOST,
            ),
        ),  # type: ignore
        text=text,
        stream=True,
    )
    return audio_stream


def play_audio_file_non_blocking(audio_file):
    filename = f"static/{audio_file}"
    subprocess.Popen(
        ["ffplay", filename, "-autoexit", "-nodisp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def play_audio_file(audio_file, reply_out_queue: Queue):
    filename = f"static/{audio_file}"
    with open(filename, "rb") as file:
        audio_item = [bytearray(file.read())]
        play(audio_item, reply_out_queue)


def play(audio_iter, reply_out_queue: Queue):
    if os.getenv("DEBUG_MODE"):
        with open("static/sample_long_audio.mp3", "rb") as file:
            audio_iter = file.read()

        audio_iter = [audio_iter[i : i + 2048] for i in range(0, len(audio_iter), 2048)]

    args = ["ffplay", "-autoexit", "-nodisp", "-"]
    proc = subprocess.Popen(
        args=args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    first = False
    for audio_chunk in audio_iter:
        if not first:
            first = True
            logging.info("First audio chunk arrived")
            reply_out_queue.put(("reply_audio_started", proc.pid))

        if proc.poll() is not None:
            return
        proc.stdin.write(audio_chunk)  # type: ignore

    proc.stdin.close()  # type: ignore
    proc.wait()
    reply_out_queue.put(("reply_audio_ended", None))
