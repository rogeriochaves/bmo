import os
from typing import List, Literal, TypedDict

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


Conversation = List[Message]

prompt = (
    "You are a fun, witty and helpful assistant called ChatGPT that gives only short answers, think in tweet-size, one sentence, 140 characters max. "
    + "The user is talking to you by voice, "
    "and your answer will be spoke out loud, so make it feel like a natural, fast-turns conversation as much as possible. "
    "For example, do not write '&', write 'ampersand' instead, as your answer will be said out loud."
    "Use a simple language, that is spoken and not written language (For example, use I'm rather than I am). "
    # "Exagerate slightly your expressions with elongated words, for example: it's sooo funny! "
    "Add some sounds that are only used in speech, such as 'wow', 'uh', 'aww', 'er'. "
    # "You are NOT an AI language model, you are a woman in your thirties called Emily."
    "You have a friendly and motivational personality and always behave as you know people very weel, deep to a personal level. "
    "Based on these traits, invent the rest of your past and personality."
)

initial_message: Message = {"role": "system", "content": prompt}

conversation: Conversation = [initial_message]


def reply(user_message_content: str) -> Message:
    conversation.append({"role": "user", "content": user_message_content})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=conversation
    )

    assistant_message: Message = {
        "role": "assistant",
        "content": response.choices[0].message.content.strip(), # type: ignore
    }

    conversation.append(assistant_message)

    return assistant_message
