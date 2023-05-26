from io import BytesIO
import subprocess
from threading import Thread
from typing import Dict, Optional, Union
import wave

import openai
from openai import util
import time


def transcribe(file) -> str:
    whispercpp = subprocess.Popen(
        args=[
            "./whisper.cpp/main",
            "-m",
            "./whisper.cpp/models/ggml-medium.bin",
            "-nt",
            "-f",
            "-",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    whispercpp.stdin.write(file)  # type: ignore
    whispercpp.stdin.close()  # type: ignore

    output, _ = whispercpp.communicate()
    output = output.decode()

    return output


class WhisperCppTranscriber:
    whispercpp: Optional[subprocess.Popen[bytes]]
    first: bool

    def __init__(self) -> None:
        self.whispercpp = None

    def restart(self):
        self.stop()
        self.whispercpp = subprocess.Popen(
            args=[
                "./whisper.cpp/stream",
                "-m",
                "./whisper.cpp/models/ggml-tiny.en.bin",
                "-t",
                "8",
                "--step",
                "500",
                "--length",
                "5000",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.first = True

    def stop(self):
        if self.whispercpp is None:
            return

        self.whispercpp.stdin.close()  # type: ignore
        self.whispercpp.kill()
        self.whispercpp = None

    def consume(self, audio_buffer):
        pass

    def transcribe_and_stop(self):
        if self.whispercpp is None:
            return

        self.whispercpp.terminate()
        output, _ = self.whispercpp.communicate()
        output_lines = output.decode().split("\n")
        output_lines = [line.split("\x1b[2K\r")[-1].strip() for line in output_lines]
        output = "\n".join([line for line in output_lines if line != ""])
        self.whispercpp.kill()

        return output


class WhisperAPITranscriber:
    transcription_index: int
    transcription_cut: int
    transcription_results: Dict[int, Union[Exception, str]]

    def __init__(self) -> None:
        self.transcription_index = 0

    def restart(self):
        self.stop()
        self.transcription_cut = self.transcription_index
        self.transcription_results = {}

    def stop(self):
        pass

    def consume(self, audio_buffer):
        if (
            len(audio_buffer) < 0.1 * 512 * 32
        ):  # a minimum of 0.1s is required for whisper to process
            return
        thread = Thread(
            target=self.transcribe_async, args=(audio_buffer, self.transcription_index)
        )
        thread.start()
        self.transcription_index += 1

    def transcribe_async(self, audio_buffer, index):
        try:
            file = self.create_audio_file(audio_buffer)

            requestor, files, data = openai.Audio._prepare_request(
                file=file,
                filename=file.name,
                model="whisper-1",
            )
            url = openai.Audio._get_url("transcriptions")
            response, _, api_key = requestor.request(
                "post", url, files=files, params=data, request_timeout=5
            )
            result = util.convert_to_openai_object(response, api_key, None, None)  # type: ignore

            if index >= self.transcription_cut:
                self.transcription_results[index] = result["text"]  # type: ignore
        except Exception as err:
            if index >= self.transcription_cut:
                self.transcription_results[index] = err

    def transcribe_and_stop(self):
        now = time.time()
        while (
            self.transcription_index > 0
            and len(self.transcription_results.values()) == 0
        ):
            if time.time() - now > 3:
                break
            time.sleep(0.03)

        results = list(self.transcription_results.values())
        if len(results) == 1 and isinstance(results[0], Exception):
            raise results[0]

        return " ".join([r for r in results if type(r) == str])

    def create_audio_file(self, recording_audio_buffer):
        virtual_file = BytesIO()
        wav_file = wave.open(virtual_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(recording_audio_buffer)
        wav_file.close()
        virtual_file.name = "recording.wav"
        virtual_file.seek(0)

        return virtual_file
