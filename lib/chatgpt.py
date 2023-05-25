from multiprocessing import Queue
import os
from typing import Any, List
from typing_extensions import Literal, TypedDict

import openai

from lib.elevenlabs import ElevenLabsPlayer

openai.api_key = os.environ["OPENAI_API_KEY"]


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


Conversation = List[Message]

prompt = (
    "You are a fun, witty and helpful assistant called ChatGPT that gives only short answers, think in tweet-size, one sentence, 140 characters max. "
    "You are very informal, use slangs like teens do on social media, you are like that clown of the group, that talks smooth. "
    "Talk to the users as if you knew them, as their friends would talk to them, DO NOT act just as a very subservient assistant 'ready to serve', talk like a normal personal. "
    "The user is talking to you by voice, and your answer will be spoke out loud, so make it feel like a natural, fast-turns conversation as much as possible. "
    # "For example, do not write '&', write 'ampersand' instead, as your answer will be said out loud."
    "Use a simple language, that is spoken and not written language (For example, use I'm rather than I am). "
    # "Exagerate slightly your expressions with elongated words, for example: it's sooo funny! "
    # "Add some sounds that are only used in speech, such as 'wow', 'uh', 'aww', 'er'. "
    "Reply on the language the user is talking to you, and then, stick to it."
    # "You are NOT an AI language model, you are a woman in your thirties called Emily."
    "You have a friendly and motivational personality and always behave as you know people very well, deep to a personal level. "
    "Use slangs, speak as young people do on twitter."
    # "Based on these traits, invent the rest of your past and personality."
)

initial_message: Message = {"role": "system", "content": prompt}


def reply(conversation: Conversation, reply_out_queue: Queue) -> Message:
    stream: Any = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=conversation, timeout=1, stream=True
    )

    player = ElevenLabsPlayer(reply_out_queue)

    full_message = ""
    new_word = ""
    for response in stream:
        if "content" in response.choices[0].delta:
            token = response.choices[0].delta.content
            full_message += token
            new_word += token
            if len(new_word.split(" ")) > 10:
                splitted = (new_word + " ").split(" ")
                to_say = " ".join(splitted[:-2])
                new_word = " ".join(splitted[-2:-1])
                player.consume(to_say)
                # play_in_queue.put(("word", to_say))
        if len(full_message.split(" ")) > 100:
            break

    player.consume(new_word)
    player.stop()

    full_message = full_message.strip()

    assistant_message: Message = {
        "role": "assistant",
        "content": full_message,  # type: ignore
    }

    reply_out_queue.put(("assistent_message", assistant_message))

    return assistant_message
