from multiprocessing import Queue
from typing import Any, Tuple
from typing_extensions import Literal


from queue import Queue

ReplyOutMsgs = Literal[
    "user_message",
    "assistent_message",
    "play_start",
    "reply_audio_started",
    "reply_audio_ended",
]

ReplyOutQueue = Queue[Tuple[ReplyOutMsgs, Any]]
