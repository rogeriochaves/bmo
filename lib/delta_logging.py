from multiprocessing.sharedctypes import Synchronized
import time
import logging
from typing import Optional

# start_time: Synchronized = Value("d", time.time())


class DeltaTimeFormatter(logging.Formatter):
    start_time: Optional[Synchronized]

    def __init__(self, fmt):
        super().__init__(fmt)
        self.start_time = None

    def format(self, record):
        if self.start_time:
            elapsed_miliseconds = round((record.created - self.start_time.value) * 1000)
            record.delta = str(elapsed_miliseconds) + "ms"
            self.start_time.value = time.time()
        else:
            record.delta = "???ms"
        return super().format(record)


yellow = "\x1b[33;20m"
red = "\x1b[31;20m"
blue = "\x1b[34;20m"
reset = "\x1b[0m"
handler = logging.StreamHandler()
LOGFORMAT = blue + "+%(delta)-7s " + reset + " %(levelname)s: %(message)s"
log_formatter = DeltaTimeFormatter(LOGFORMAT)
handler.setFormatter(log_formatter)
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)
