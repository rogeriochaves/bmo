import multiprocessing
import os
from queue import Empty, Queue
import subprocess
from threading import Thread
from typing import Dict, Iterator, List, Union
from typing_extensions import Literal
from elevenlabs import generate, Voice, VoiceSettings
from lib.delta_logging import logging

eleven_labs_api_key = os.environ["ELEVEN_LABS_API_KEY"]

VOICE_SETTINGS_STABILITY = 0.75
VOICE_SETTINGS_SIMILARITY_BOOST = 0.75
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # pNInz6obpgDQGcFmaJgB

logger = logging.getLogger()


class ElevenLabsPlayer:
    ffplay: subprocess.Popen[bytes]
    reply_out_queue: multiprocessing.Queue
    word_index: int
    audio_chunks: Dict[int, List[Union[bytes, Literal["done"]]]]
    playing_index: int
    requested_to_stop: bool
    local_queue: Queue

    def __init__(self, reply_out_queue: multiprocessing.Queue) -> None:
        self.reply_out_queue = reply_out_queue
        self.requested_to_stop = False
        self.local_queue = Queue()
        self.start()

    def start(self):
        self.word_index = 0
        self.playing_index = 0
        self.audio_chunks = {}
        self.ffplay = subprocess.Popen(
            args=["ffplay", "-autoexit", "-nodisp", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )

    def stop(self):
        self.requested_to_stop = True
        while True:
            action, data = self.local_queue.get(block=True)
            self.reply_out_queue.put((action, data))
            if action == "reply_audio_ended":
                break

    def _stop(self):
        self.ffplay.stdin.close()  # type: ignore
        self.ffplay.wait()
        self.local_queue.put(("reply_audio_ended", None))

    def consume(self, word: str):
        self.audio_chunks[self.word_index] = []
        thread = Thread(target=self.generate_async, args=(word, self.word_index))
        thread.start()
        self.word_index += 1
        try:
            self.reply_out_queue.put(self.local_queue.get(block=False))
        except Empty:
            pass

    def generate_async(self, word: str, index: int):
        audio_stream: Iterator[bytes] = generate(
            api_key=eleven_labs_api_key,
            model="eleven_multilingual_v1",
            voice=Voice(
                voice_id=VOICE_ID,
                settings=VoiceSettings(
                    stability=VOICE_SETTINGS_STABILITY,
                    similarity_boost=VOICE_SETTINGS_SIMILARITY_BOOST,
                ),
            ),  # type: ignore
            text=self.speechify(word),
            stream=True,
        )

        for chunk_index, audio_chunk in enumerate(audio_stream):
            if index == 0 and chunk_index == 0:
                logging.info("First audio chunk arrived")
                self.local_queue.put(("reply_audio_started", self.ffplay.pid))

            self.audio_chunks[index].append(audio_chunk)

            if index == self.playing_index:
                self.play_next_chunks(keep=2)

        self.audio_chunks[index] = self.audio_chunks[index][
            :-2  # drop the last few frames to trim the ending silence of elevenlabs
        ]
        self.audio_chunks[index].append("done")
        self.play_next_chunks()

    def play_next_chunks(self, keep=0):
        if self.ffplay.poll() is not None:
            return

        if self.playing_index not in self.audio_chunks:
            if self.requested_to_stop:
                self._stop()
            return

        while len(self.audio_chunks[self.playing_index]) > keep:
            audio_chunk = self.audio_chunks[self.playing_index].pop(0)
            if audio_chunk == "done":
                self.playing_index += 1
                self.play_next_chunks()
                break

            self.ffplay.stdin.write(audio_chunk)  # type: ignore

    def speechify(self, word: str):
        return word.replace("#", "hashtag ")


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
    reply_out_queue.put(("reply_audio_ended", None))


# def eleven_labs_player_process(in_queue: Queue, reply_out_queue: Queue):
#     player = ElevenLabsPlayer(reply_out_queue)

#     while True:
#         action, word = in_queue.get()
#         if action == "stop":
#             player.stop()
#             break
#         else:
#             player.consume(word)


# def play_with_piper(in_queue: Queue, reply_out_queue: Queue):
#     args = [
#         "bin/piper-macos-m1",
#         "--model",
#         "./models/voice-en-us-libritts-high/en-us-libritts-high.onnx",
#         "-f",
#         "-",
#     ]
#     piper_proc = subprocess.Popen(
#         args=args,
#         stdin=subprocess.PIPE,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.DEVNULL,
#     )

#     ffplay_proc = subprocess.Popen(
#         args=["ffplay", "-autoexit", "-nodisp", "-"],
#         stdin=piper_proc.stdout,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )

#     first = False
#     while True:
#         action, word = in_queue.get()
#         if action == "stop":
#             break

#         if not first:
#             first = True
#             logging.info("First audio chunk arrived")
#             reply_out_queue.put(("reply_audio_started", piper_proc.pid))

#         if piper_proc.poll() is not None:
#             return
#         piper_proc.stdin.write(word.encode())  # type: ignore

#     piper_proc.stdin.close()  # type: ignore
#     piper_proc.wait()
#     ffplay_proc.wait()
#     reply_out_queue.put(("reply_audio_ended", None))
