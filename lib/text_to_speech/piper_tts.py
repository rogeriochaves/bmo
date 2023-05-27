import multiprocessing
from multiprocessing import Queue
import select
import subprocess
from threading import Thread
from lib.delta_logging import logging

logger = logging.getLogger()


class PiperTTS:
    min_words = 2
    piper: subprocess.Popen
    ffplay: subprocess.Popen
    from_piper_to_ffplay: Thread
    reply_out_queue: multiprocessing.Queue
    local_queue: Queue
    requested_to_stop: bool

    def __init__(self, reply_out_queue: multiprocessing.Queue) -> None:
        self.reply_out_queue = reply_out_queue
        self.start()

    def start(self):
        self.requested_to_stop = False
        self.piper = subprocess.Popen(
            [
                "./piper/piper/piper",
                "--model",
                "./piper/en-us-ryan-medium.onnx",
                "--output_raw",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.ffplay = subprocess.Popen(
            [
                "ffplay",
                "-probesize",
                "8192",
                "-f",
                "s16le",
                "-ar",
                "22050",
                "-ac",
                "1",
                "-nodisp",
                "-autoexit",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        self.from_piper_to_ffplay = Thread(target=self.play_as_available)
        self.from_piper_to_ffplay.start()

    def wait_to_finish(self):
        self.stop()

    def stop(self):
        self.requested_to_stop = True
        remaining, _ = self.piper.communicate()
        if remaining is not None:
            self.ffplay.stdin.write(remaining)  # type: ignore
        self.ffplay.stdin.close()  # type: ignore
        self.ffplay.wait()

        self.reply_out_queue.put(("reply_audio_ended", None))

    def consume(self, word: str):
        if word == "":
            return

        self.piper.stdin.write((word + "\n").encode("utf-8"))  # type: ignore
        self.piper.stdin.flush()  # type: ignore

    def play_as_available(self):
        first = True
        while not self.requested_to_stop:
            # Wait for data to become available
            ready_to_read, _, _ = select.select([self.piper.stdout.fileno()], [], [])  # type: ignore
            for stream in ready_to_read:
                if self.requested_to_stop:
                    return
                if stream == self.piper.stdout.fileno():  # type: ignore
                    output = self.piper.stdout.read1(512 * 32)  # type: ignore
                    if output:
                        if first:
                            logger.info("First audio chunk arrived")
                            self.reply_out_queue.put(
                                ("reply_audio_started", self.ffplay.pid)
                            )
                            first = False
                        self.ffplay.stdin.write(output)  # type: ignore
                    else:
                        # No more output, break the loop
                        return
