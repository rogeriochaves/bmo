import os
import subprocess
from elevenlabs import generate, Voice, VoiceSettings
from lib.delta_logging import logging

eleven_labs_api_key = os.environ["ELEVEN_LABS_API_KEY"]

VOICE_SETTINGS_STABILITY = 0.3
VOICE_SETTINGS_SIMILARITY_BOOST = 0.75
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

logger = logging.getLogger()


def text_to_speech(text: str):
    audio_stream = generate(
        api_key=eleven_labs_api_key,
        voice=Voice(
            voice_id=VOICE_ID,
            settings=VoiceSettings(
                stability=VOICE_SETTINGS_STABILITY,
                similarity_boost=VOICE_SETTINGS_SIMILARITY_BOOST,
            ),
        ),
        text=text,
        stream=True,
    )
    return audio_stream


def play_audio_file_non_blocking(beep_file):
    filename = f"static/{beep_file}"
    subprocess.Popen(
        ["ffplay", filename, "-autoexit", "-nodisp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def play(audio_iter):
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
        proc.stdin.write(audio_chunk)

    proc.stdin.close()
    proc.wait()
