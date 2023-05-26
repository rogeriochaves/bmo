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

VOICE_SETTINGS_STABILITY = 1
VOICE_SETTINGS_SIMILARITY_BOOST = 0.75
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"  # pNInz6obpgDQGcFmaJgB

logger = logging.getLogger()


class SayPlayer:
    reply_out_queue: multiprocessing.Queue
    word_index: int
    playing_index: int
    local_queue: Queue

    def __init__(self, reply_out_queue: multiprocessing.Queue) -> None:
        self.reply_out_queue = reply_out_queue
        self.local_queue = Queue()
        self.word_index = 0
        self.playing_index = 0

    def start(self):
        pass

    def stop(self):
        self.local_queue = Queue()
        while True:
            action, data = self.local_queue.get(block=True)
            self.reply_out_queue.put((action, data))
            if action == "reply_audio_ended":
                break

    def consume(self, word: str):
        if self.word_index == 0:
            self.reply_out_queue.put(("reply_audio_started", -1))
        thread = Thread(target=self.generate_async, args=(word, self.word_index))
        thread.start()
        self.word_index += 1

    def generate_async(self, word: str, index: int):
        while self.playing_index != index:
            pass
        subprocess.call(["say", word, "-r", "200"])
        self.playing_index += 1
        if self.playing_index == self.word_index:
            self.local_queue.put(("reply_audio_ended", None))


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

    def request_to_stop(self):
        self.requested_to_stop = True
        while True:
            action, data = self.local_queue.get(block=True)
            self.reply_out_queue.put((action, data))
            if action == "reply_audio_ended":
                break

    def terminate(self):
        self.ffplay.stdin.close()  # type: ignore
        self.ffplay.wait()
        self.local_queue.put(("reply_audio_ended", None))

    def consume(self, word: str):
        if word == "":
            return
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
                self.play_next_chunks()

        self.audio_chunks[index].append("done")
        self.play_next_chunks()

    def play_next_chunks(self):
        if self.playing_index not in self.audio_chunks:
            if self.requested_to_stop:
                self.terminate()
            return

        if self.ffplay.poll() is not None:
            return

        while len(self.audio_chunks[self.playing_index]) > 0:
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
