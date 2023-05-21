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
    return util.convert_to_openai_object(response, api_key, None, None) # type: ignore
