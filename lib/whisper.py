import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing_extensions import TypedDict
import openai
from openai import util


class TranscriptionResult(TypedDict):
    text: str


def transcribe(model, file, **params) -> TranscriptionResult:
    requestor, files, data = openai.Audio._prepare_request(
        file=file,
        filename=file.name,
        model=model,
        **params,
    )
    url = openai.Audio._get_url("transcriptions")
    response, _, api_key = requestor.request(
        "post", url, files=files, params=data, request_timeout=5
    )
    result = util.convert_to_openai_object(response, api_key, None, None)  # type: ignore

    return result  # type: ignore


def async_transcribe(executor: ThreadPoolExecutor, model, file, **params):
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(executor, transcribe, model, file, **params)
    return future
