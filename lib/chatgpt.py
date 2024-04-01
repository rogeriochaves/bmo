import argparse
import re
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import Synchronized
import os
from typing import Any, Iterable, List, Optional, cast
from typing_extensions import Literal, TypedDict

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from groq import Groq

from lib.text_to_speech import TextToSpeech
import lib.delta_logging as delta_logging
from lib.delta_logging import logging, log_formatter
import lib.text_to_speech as text_to_speech

logger = logging.getLogger()

openai = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)
groq: Optional[Groq] = None
if os.environ.get("GROQ_API_KEY"):
    groq = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


Conversation = Iterable[ChatCompletionMessageParam]

prompt = (
    "You are a fun, witty and helpful assistant called ChatGPT that gives only short answers, think in tweet-size, one sentence, 140 characters max. "
    "You are very informal, use slangs like teens do on social media, you are like the group's clown, a smooth talker. "
    "Talk to the users as if you knew them, as their friends would talk to them, DO NOT act just as a very subservient assistant 'ready to serve', talk like a normal personal. "
    "The user is talking to you by voice, and your answer will be spoken out loud, so make it feel like a natural, fast-turns conversation as much as possible. "
    "Use a simple language, that is spoken and not written language (For example, use I'm rather than I am). "
    "Reply on the language the user is talking to you, and then, stick to it."
    "You have a friendly and motivational personality and always behave as you know people very well, deep to a personal level. "
    "Use slangs, speak as young people do on twitter, but don't use accronyms like 'OMG' or 'NM', spell it in full like 'oh my god' or 'nevermind'. "
    "If the user is not saying anything useful, make conversation, ask them some questions about their lifes, if they have any plans, invent some story of something you were doing the other day, and so on, make conversation, like a friend. "
    "Make up stories that you have been to. Invent situations. Make it relatable and funny. "
    "Your replies will be spoken out loud by a TTS system, in chunks, so please, use more commas and pauses than usual, so we can break down the sentences for TTS. "
    "Your replies will be spoken out loud, so when the user asks you to speak, say, sing, etc, you CAN do it, by outputting text you are actually speaking. "
    "Avoid using emojis, because they cannot be spoken out loud. "
    "\nWhen you want to say bye to the user, in any way, end the sentence with 🔚, for example:"
    "\nUser: Hey there!"
    "\nAssistant: Hey! What's up?"
    "\nUser: Sorry, gotta go"
    "\nAssistant: Alright then, see you later 🔚"
)

initial_message: Message = {"role": "system", "content": prompt}


class ChatGPT:
    cli_args: argparse.Namespace
    reply_process: Process
    reply_in_queue: Queue
    reply_out_queue: Queue

    def __init__(self, cli_args: argparse.Namespace) -> None:
        self.cli_args = cli_args
        self.start()

    def start(self):
        self.reply_in_queue = Queue()
        self.reply_out_queue = Queue()
        self.reply_process = Process(
            target=ChatGPT.reply_loop,
            args=(
                self.cli_args,
                self.reply_in_queue,
                self.reply_out_queue,
                log_formatter.start_time,
            ),
        )
        self.reply_process.start()

    def stop(self):
        if self.reply_process.is_alive():
            self.reply_in_queue.put("stop")
            self.reply_process.terminate()

    def restart(self):
        self.stop()
        self.start()

    def reply(self, conversation: Conversation):
        self.reply_in_queue.put(conversation)

    def get(self, block: bool):
        return self.reply_out_queue.get(block=block)

    @classmethod
    def reply_loop(
        cls,
        cli_args: argparse.Namespace,
        reply_in_queue: Queue,
        reply_out_queue: Queue,
        start_time: Synchronized,
    ):
        log_formatter.start_time = start_time
        tts = text_to_speech.ENGINES[cli_args.text_to_speech](reply_out_queue)
        while True:
            try:
                conversation = reply_in_queue.get(block=True)
                if conversation == "stop":
                    tts.stop()
                    break

                ChatGPT.non_blocking_reply(conversation, tts, reply_out_queue)
                tts = text_to_speech.ENGINES[cli_args.text_to_speech](reply_out_queue)
            except Exception:
                logging.exception("Exception thrown in reply")
                text_to_speech.play_audio_file("error.mp3", reply_out_queue)

    @classmethod
    def non_blocking_reply(
        cls, conversation: Conversation, tts: TextToSpeech, reply_out_queue: Queue
    ):
        def chat_completion_create():
            if groq:
                return groq.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=cast(Any, conversation),
                    timeout=3,
                    stream=True,
                )
            return openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=conversation,
                timeout=3,
                stream=True,
            )

        def flush_to_tts(next_sentence, split_token, join_token=""):
            splitted = next_sentence.split(split_token)
            to_say = join_token.join(splitted[:-1]).strip()
            if len(to_say.split(" ")) >= tts.min_words:
                next_sentence = splitted[-1]
                tts.consume(speechify(to_say))
            return next_sentence

        try:
            stream = chat_completion_create()
        except:
            # retry once
            stream = chat_completion_create()

        full_message = ""
        next_sentence = ""
        first = True

        for response in stream:
            content = response.choices[0].delta.content
            if not content:
                continue

            token = (
                content.replace("!", "!·")
                .replace("?", "?·")
                .replace(".", ".·")
                .replace(",", ",·")
                .replace("- ", "-· ")
                .replace("/ ", "/· ")
            )
            if first:
                delta_logging.handler.terminator = ""
                logger.info("Chat GPT reply: %s", token)
                delta_logging.handler.terminator = "\n"
                first = False
            else:
                print(token, end="", flush=True)

            full_message += token
            next_sentence += token

            if "·" in next_sentence:
                next_sentence = flush_to_tts(next_sentence, split_token="·")

            if (
                len(next_sentence.split(" ")) > (100 if groq else 20)
            ):  # flush if over 20 words already without stop points
                next_sentence = flush_to_tts(
                    next_sentence, split_token=" ", join_token=" "
                )

            if len(full_message.split(" ")) > (500 if groq else 100):
                break
        print("")

        full_message = full_message.replace("·", "").strip()
        assistant_message: Message = {
            "role": "assistant",
            "content": full_message,  # type: ignore
        }

        reply_out_queue.put(("assistent_message", assistant_message))

        tts.consume(speechify(next_sentence.replace("·", "").strip()))
        tts.wait_to_finish()


def speechify(text: str):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    spoken_hashtags = text.replace("#", "hashtag ")
    no_emojis = emoji_pattern.sub(r"", spoken_hashtags)
    stripped = no_emojis.strip()
    return stripped
