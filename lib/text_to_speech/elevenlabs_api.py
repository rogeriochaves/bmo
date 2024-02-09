import os
from queue import Empty, Queue
import subprocess
import multiprocessing
from threading import Thread, Semaphore
from lib.delta_logging import logging
from typing import Dict, Iterator, List, Union
from typing_extensions import Literal
from elevenlabs import generate, Voice, VoiceSettings

logger = logging.getLogger()

eleven_labs_api_key = "cb07e6d99a69e1caf4cca7b29e99e9c6"

VOICE_SETTINGS_STABILITY = 1
VOICE_SETTINGS_SIMILARITY_BOOST = 0.75
VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # pNInz6obpgDQGcFmaJgB


class ElevenLabsAPI:
    min_words = 2
    ffplay: subprocess.Popen
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
        self.semaphore = Semaphore(1) 
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

    def wait_to_finish(self):
        self.requested_to_stop = True
        while True:
            action, data = self.local_queue.get(block=True)
            self.reply_out_queue.put((action, data))
            if action == "reply_audio_ended":
                break

    def stop(self):
        self.ffplay.stdin.close()  # type: ignore
        self.ffplay.wait()
        self.local_queue.put(("reply_audio_ended", None))

    def consume(self, word: str):
        if word == "":
            return
        self.audio_chunks[self.word_index] = []
        thread = Thread(target=self.generate_async, args=(word, self.word_index))
        thread.start()
        #self.generate_async(word, self.word_index)
        self.word_index += 1
        try:
            self.reply_out_queue.put(self.local_queue.get(block=False))
        except Empty:
            pass

    def generate_async(self, word: str, index: int):
        try:
            self.semaphore.acquire()  # Acquire semaphore before proceeding
            audio_stream: Iterator[bytes] = generate(
                api_key=eleven_labs_api_key,
                model="eleven_multilingual_v2",
                voice=Voice(
                    voice_id=VOICE_ID,
                    settings=VoiceSettings(
                        stability=VOICE_SETTINGS_STABILITY,
                        similarity_boost=VOICE_SETTINGS_SIMILARITY_BOOST,
                    ),
                ),
                text=word,
                stream=True,
            )

            for chunk_index, audio_chunk in enumerate(audio_stream):
                if index == 0 and chunk_index == 0:
                    logger.info("First audio chunk arrived")
                    self.local_queue.put(("reply_audio_started", self.ffplay.pid))
                
                self.audio_chunks[index].append(audio_chunk)

                if index == self.playing_index:
                    self.play_next_chunks()

            self.audio_chunks[index].append("done")
            self.play_next_chunks()
        finally:
            self.semaphore.release() 

    def play_next_chunks(self):

        if self.playing_index not in self.audio_chunks:
            if self.requested_to_stop:
                self.stop()
            return

        if self.ffplay.poll() is not None:
            return
        while len(self.audio_chunks[self.playing_index]) > 0:
            audio_chunk = self.audio_chunks[self.playing_index].pop(0)
            if audio_chunk == "done":
                self.playing_index += 1
                print("gonderrrrrrrrrrrrrrrrrr")
                
                self.play_next_chunks()
                break

            self.ffplay.stdin.write(audio_chunk)  # type: ignore
