from io import BytesIO
from multiprocessing import Queue
import os
import struct
import subprocess
import wave
from elevenlabs import generate, Voice, VoiceSettings
from lib.delta_logging import logging
import ffmpeg

from lib.interruption_detection import InterruptionDetection

eleven_labs_api_key = os.environ["ELEVEN_LABS_API_KEY"]

VOICE_SETTINGS_STABILITY = 0.75
VOICE_SETTINGS_SIMILARITY_BOOST = 0.75
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # pNInz6obpgDQGcFmaJgB

logger = logging.getLogger()


def text_to_speech(text: str):
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


def play_audio_file_non_blocking(beep_file):
    filename = f"static/{beep_file}"
    subprocess.Popen(
        ["ffplay", filename, "-autoexit", "-nodisp"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def play(audio_iter, reply_out_queue: Queue):
    args = ["ffplay", "-autoexit", "-nodisp", "-"]
    proc = subprocess.Popen(
        args=args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )

    first = False
    reply_buffer = bytearray()
    reply_out_queue.put(("play_start", proc.pid))

    for audio_chunk in audio_iter:
        if not first:
            first = True
            logging.info("First audio chunk arrived")

        if proc.poll() is not None:
            return
        proc.stdin.write(audio_chunk)  # type: ignore

        reply_buffer.extend(audio_chunk)
        # TODO: optimize, send only the first buffer then checkpoints
        if len(reply_buffer) >= 512:
            pcm_bytes = mp3_to_pcm(reply_buffer)
            pcm_ints = list(struct.unpack("h" * (len(pcm_bytes) // 2), pcm_bytes))
            reply_out_queue.put(("reply_audio", pcm_ints))

    proc.stdin.close()  # type: ignore
    proc.wait()


def mp3_to_pcm(mp3_bytes):
    try:
        # Convert the audio file to PCM using ffmpeg
        wav_bytes, err = (
            ffmpeg.input("pipe:0", format="mp3")
            .output("pipe:1", format="wav", acodec="pcm_s16le", ac=1, ar="16k")
            .run(input=mp3_bytes, capture_stdout=True, capture_stderr=True)
        )

        wav_file = wave.open(BytesIO(wav_bytes), "rb")
        frames = wav_file.readframes(wav_file.getnframes())
        wav_file.close()
        return frames
    except ffmpeg.Error as e:
        print("FFmpeg error:")
        print(e.stderr.decode())
        raise e
