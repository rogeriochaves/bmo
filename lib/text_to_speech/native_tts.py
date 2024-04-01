import multiprocessing
from multiprocessing import Queue
import platform
import subprocess
from threading import Thread
from typing import Optional
from lib.delta_logging import logging
from queue import Empty

logger = logging.getLogger()


class NativeTTS:
    min_words = 2
    reply_in_queue: multiprocessing.Queue
    reply_out_queue: multiprocessing.Queue
    word_index: int
    playing_index: int
    local_queue: Queue
    current_process: Optional[subprocess.Popen] = None

    def __init__(
        self,
        reply_in_queue: multiprocessing.Queue,
        reply_out_queue: multiprocessing.Queue,
    ) -> None:
        self.reply_in_queue = reply_in_queue
        self.reply_out_queue = reply_out_queue
        self.local_queue = Queue()
        self.word_index = 0
        self.playing_index = 0

    def start(self):
        pass

    def wait_to_finish(self):
        self.local_queue = Queue()
        while True:
            try:
                outside_action = self.reply_in_queue.get(block=False)
                if outside_action == "stop":
                    self.stop()
                    break
            except Empty:
                pass

            try:
                action, data = self.local_queue.get(block=False)
                self.reply_out_queue.put((action, data))
                if action == "reply_audio_ended":
                    break
            except Empty:
                pass

    def stop(self):
        if self.current_process:
            self.current_process.terminate()
            logger.info("Subprocess terminated")
        self.current_process = None

    def consume(self, word: str):
        if word == "":
            return
        if self.word_index == 0:
            logger.info("First audio chunk arrived")
            self.reply_out_queue.put(("reply_audio_started", -1))
        thread = Thread(target=self.generate_async, args=(word, self.word_index))
        thread.start()
        self.word_index += 1

    def generate_async(self, word: str, index: int):
        while self.playing_index != index:
            pass
        # Weirdly say sometimes hang and never return, so we use subprocess.call instead for now
        # cmd = ["say", word] if platform.system() == "Darwin" else ["espeak-ng", word]
        # self.current_process = subprocess.Popen(
        #     cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        # )
        # try:
        #     self.current_process.wait(timeout=15)
        # except subprocess.TimeoutExpired:
        #     logger.info(cmd[0] + " subprocess timed out")
        #     self.current_process.terminate()
        if platform.system() == "Darwin":
            subprocess.call(["say", word])
        else:
            subprocess.call(["espeak-ng", word])
        self.playing_index += 1
        if self.playing_index == self.word_index:
            self.local_queue.put(("reply_audio_ended", None))
