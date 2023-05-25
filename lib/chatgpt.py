from multiprocessing import Queue
import os
from typing import Any, List
from typing_extensions import Literal, TypedDict

import openai

from lib.elevenlabs import ElevenLabsPlayer, SayPlayer
import lib.delta_logging as delta_logging
from lib.delta_logging import logging

logger = logging.getLogger()

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
    "Use slangs, speak as young people do on twitter, but don't use accronyms like OMG or NM, spell it in full like oh my god or nevermind."
    # "Based on these traits, invent the rest of your past and personality."
    "Your replies will be spoken out loud by a TTS system, in chunks, so please, use more commas and pauses than usual, so we can break down the sentences for TTS"
    # "Your replies will be spoken out loud by a TTS system, in chunks, so please, very important, after every sentence, comma, period, -, or really any chunk that seems like a good chunk to send to TTS to be spoken out loud, put a · character. For example:"
    # "\nUser: hello there"
    # "\nAssistant: Yo!· Hey there.· What's up?·"
    # "\n\nAnother example:"
    # "\nAssistant: Mario was on a mission,· to save Princess Peach from Bowser,· dodging Goombas,· collecting coins,· and power-ups to gain strength,· until he finally reached the castle where he faced Bowser in an epic battle,· and emerged victorious,· saving Princess Peach and the Mushroom Kingdom!·"
    # "\n\nAnother example:"
    # "\nAssistant: Hey!· Not much,· just hanging out and ready to assist you -· this is what I like to do.· What can I help you with today?·"
    # "\n\nKeep adding · this is very important, do not ever forget, for the whole conversation, add · after every comma, period, exclamation, question mark, for example: ', - . ? !' should become ',· -· ?· !·'. The reason is I want to use the special token · to break down the sentences using python to send to the TTS system"
)

initial_message: Message = {"role": "system", "content": prompt}


def reply(conversation: Conversation, reply_out_queue: Queue) -> Message:
    stream: Any = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=conversation, timeout=1, stream=True
    )

    player = SayPlayer(reply_out_queue)

    full_message = ""
    next_sentence = ""
    first = True

    for response in stream:
        if "content" not in response.choices[0].delta:
            continue

        token = (
            response.choices[0]
            .delta.content.replace("!", "!·")
            .replace("?", "?·")
            .replace(".", ".·")
            .replace(",", ",·")
            .replace("- ", "-· ")
        )
        if first:
            delta_logging.handler.terminator = ""
            logger.info("Chat GPT started replying: %s", token)
            delta_logging.handler.terminator = "\n"
            first = False
        else:
            print(token, end="", flush=True)

        full_message += token
        next_sentence += token

        if "·" in next_sentence:
            splitted = next_sentence.split("·")
            to_say = "".join(splitted[:-1]).strip()
            next_sentence = splitted[-1]

            player.consume(to_say)

        if len(full_message.split(" ")) > 100:
            break
    print("")

    player.consume(next_sentence.replace("·", "").strip())
    player.stop()

    full_message = full_message.replace("·", "").strip()

    assistant_message: Message = {
        "role": "assistant",
        "content": full_message,  # type: ignore
    }

    reply_out_queue.put(("assistent_message", assistant_message))

    return assistant_message
