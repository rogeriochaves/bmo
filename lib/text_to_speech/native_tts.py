import multiprocessing
from multiprocessing import Queue
import subprocess
from threading import Thread
from lib.delta_logging import logging

logger = logging.getLogger()


class NativeTTS:
    min_words = 2
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

    def request_to_stop(self):
        self.local_queue = Queue()
        while True:
            action, data = self.local_queue.get(block=True)
            self.reply_out_queue.put((action, data))
            if action == "reply_audio_ended":
                break

    def consume(self, word: str):
        if self.word_index == 0:
            logger.info("First audio chunk arrived")
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
