import time
import logging

class DeltaTimeFormatter(logging.Formatter):
    start_time: float

    def __init__(self, fmt):
        super().__init__(fmt)
        self.start_time = time.time()

    def format(self, record):
        elapsed_miliseconds = round((record.created - self.start_time) * 1000)
        record.delta = str(elapsed_miliseconds) + "ms"
        self.start_time = time.time()
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
