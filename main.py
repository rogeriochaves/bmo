import argparse
from multiprocessing import Value
from multiprocessing.sharedctypes import Synchronized
from threading import Thread
import time
from dotenv import load_dotenv  # has to be the first import

load_dotenv()
from lib.delta_logging import logging, red, reset, log_formatter  # has to be the second
from queue import Empty
from typing import Any, List, Optional
from typing_extensions import Literal
from lib.interruption_detection import InterruptionDetection
from lib.porcupine import wakeup_keywords
from lib.utils import calculate_volume
from lib.chatgpt import ChatGPT, Conversation, Message, initial_message
import lib.text_to_speech as text_to_speech
import lib.speech_recognition as speech_recognition
from lib.speech_recognition import SpeechRecognition
import os
import struct
import pvporcupine
from pvrecorder import PvRecorder
import openai

picovoice_access_key = os.environ.get("PICOVOICE_ACCESS_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]

logger = logging.getLogger()

frame_length = 512
buffer_size_when_not_listening = frame_length * 32 * 2  # keeps 2s of audio
buffer_size_on_active_listening = frame_length * 32 * 60  # keeps 60s of audio
sample_rate = 16000  # sample rate for Porcupine is fixed at 16kHz
silence_threshold = 300  # maybe need to be adjusted
silence_limit = 0.5 * 32  # 0.5 seconds of silence
speaking_minimum = 0.3 * 32  # 0.3 seconds of speaking
silence_time_to_standby = (
    10 * 32
)  # goes back to wakeup word checking after 10s of silence
restart_recorder_every = 60  # restarts mic recorder every 60s when in standby because sometimes it get stuck in buffer overflow error


RecordingState = Literal[
    "waiting_for_wakeup",
    "waiting_for_silence",
    "start_reply",
    "replying",
]


class AudioRecording:
    state: RecordingState
    conversation: Conversation = [initial_message]

    porcupine: Optional[pvporcupine.Porcupine]
    recorder: PvRecorder
    recorder_started_at: float
    cli_args: argparse.Namespace

    silence_frame_count: int
    speaking_frame_count: int
    recording_audio_buffer: bytearray

    chat_gpt: ChatGPT
    interruption_detection: InterruptionDetection
    speech_recognition: SpeechRecognition

    def __init__(self, recorder: PvRecorder, cli_args: argparse.Namespace) -> None:
        self.recorder = recorder
        self.recorder_started_at = time.time()
        self.cli_args = cli_args
        self.recording_audio_buffer = bytearray()
        self.speaking_frame_count = 0
        self.chat_gpt = ChatGPT(cli_args)
        self.interruption_detection = InterruptionDetection()
        self.speech_recognition = speech_recognition.ENGINES[
            cli_args.speech_recognition
        ]()
        self.speech_recognition.restart()
        self.reset("waiting_for_silence")

        if picovoice_access_key:
            self.porcupine = pvporcupine.create(
                access_key=picovoice_access_key,
                keyword_paths=wakeup_keywords(),
            )
        else:
            self.porcupine = None

    def reset(self, state):
        self.recorder.start()
        self.state = state

        self.silence_frame_count = 0

        if state == "waiting_for_silence":
            self.interruption_detection.reset()
        elif state == "replying":
            pass
        elif state == "start_reply":
            pass
        elif state == "waiting_for_wakeup":
            text_to_speech.play_audio_file_non_blocking("beep_standby.mp3")
            self.silence_frame_count = 0
            self.speaking_frame_count = 0
            self.chat_gpt.stop()
            self.speech_recognition.stop()
            self.interruption_detection.stop()

    def stop(self):
        self.recorder.stop()
        self.chat_gpt.stop()
        self.interruption_detection.stop()
        self.speech_recognition.stop()
        if self.porcupine:
            self.porcupine.delete()

    def transcribe_buffer(self):
        self.speech_recognition.consume(self.recording_audio_buffer)
        self.recording_audio_buffer = self.recording_audio_buffer[
            -max(frame_length * 32 * 3, 0) :
        ]

    def next_frame(self):
        pcm = self.recorder.read()

        self.recording_audio_buffer.extend(struct.pack("h" * len(pcm), *pcm))
        self.drop_early_recording_audio_frames()

        if self.state == "waiting_for_wakeup":
            self.waiting_for_wakeup(pcm)

        elif self.state == "waiting_for_silence":
            self.waiting_for_silence(pcm)

        elif self.state == "start_reply":
            start_reply_thread = Thread(target=self.start_reply_async)
            start_reply_thread.start()
            self.reset("replying")

        elif self.state == "replying":
            self.replying_loop(pcm)

    def start_reply_async(self):
        transcription = self.speech_recognition.transcribe_and_stop()
        if len(transcription.strip()) == 0:
            logger.info("Transcription too small, probably a mistake, bailing out")
            self.reset("waiting_for_silence")
            return

        if self.state != "replying":
            return  # probably got interrupted

        user_message: Message = {"role": "user", "content": transcription}
        self.conversation.append(user_message)
        self.recording_audio_buffer = self.recording_audio_buffer[-frame_length:]

        self.chat_gpt.reply(self.conversation)

    def is_silence(self, pcm):
        rms = calculate_volume(pcm)
        return rms < silence_threshold

    def drop_early_recording_audio_frames(self):
        if len(self.recording_audio_buffer) > (
            buffer_size_when_not_listening
            if self.state == "waiting_for_wakeup"
            else buffer_size_on_active_listening
        ):
            self.recording_audio_buffer = self.recording_audio_buffer[
                frame_length:
            ]  # drop early frames to keep just most recent audio

    def waiting_for_wakeup(self, pcm: List[Any]):
        if not self.porcupine:
            self.reset("waiting_for_silence")
            return
        if time.time() - self.recorder_started_at > restart_recorder_every:
            self.recorder.stop()
            self.recorder.start()
            self.recorder_started_at = time.time()

        print(f"⚪️ Waiting for wake up word...", end="\r", flush=True)
        trigger = self.porcupine.process(pcm)
        if trigger >= 0:
            logger.info("Detected wakeup word #%s", trigger)
            text_to_speech.play_audio_file_non_blocking("beep_wakeup.mp3")
            self.chat_gpt.restart()
            self.speech_recognition.restart()
            self.interruption_detection.start()
            self.transcribe_buffer()
            self.state = "start_reply"

    def waiting_for_silence(self, pcm: List[Any]):
        is_silence = self.is_silence(pcm)
        emoji = "🔈" if is_silence else "🔊"
        print(f"🔴 {red}Listening... {emoji} {reset}", end="\r", flush=True)

        if is_silence:
            self.silence_frame_count += 1
            if (
                self.speaking_frame_count < speaking_minimum
                and self.silence_frame_count >= silence_limit * 2
            ):
                self.speaking_frame_count = 0
        else:
            # Cut all empty audio from before to make it smaller
            if self.speaking_frame_count == 0:
                self.recording_audio_buffer = self.recording_audio_buffer[
                    -frame_length * 4 :
                ]
            self.speaking_frame_count += 1
            self.silence_frame_count = 0

        transcription_flush_step = 1 * 32  # 1s of audio
        if (
            self.speaking_frame_count > 0
            and (self.silence_frame_count + self.speaking_frame_count)
            % transcription_flush_step
            == 0
        ):
            self.transcribe_buffer()

        if (
            self.silence_frame_count >= silence_limit
            and self.speaking_frame_count >= speaking_minimum
        ):
            logger.info("Detected silence a while after speaking, giving a reply")
            self.transcribe_buffer()
            self.state = "start_reply"

        if (
            self.porcupine is not None
            and self.silence_frame_count >= silence_time_to_standby
        ):
            logger.info("Long silence time, going back to waiting for the wakeup word")
            self.recorder.stop()
            text_to_speech.play_audio_file("byebye.mp3")
            self.reset("waiting_for_wakeup")

    def replying_loop(self, pcm: List[Any]):
        try:
            (action, data) = self.chat_gpt.get(block=False)
            if action == "assistent_message":
                self.conversation.append(data)
            elif action == "reply_audio_started":
                self.silence_frame_count = 0
                self.speaking_frame_count = 0
                self.interruption_detection.start_reply_interruption_check(data)
                self.speech_recognition.restart()
            elif action == "reply_audio_ended":
                self.interruption_detection.stop()
                if "🔚" in self.conversation[-1]["content"]:
                    self.reset("waiting_for_wakeup")
                    return
        except Empty:
            pass

        if self.interruption_detection.is_done():
            self.recording_audio_buffer = self.recording_audio_buffer[
                -frame_length * 2 :
            ]  # Capture the last couple frames for better follow up after assistant reply
            self.speaking_frame_count = 0
            self.reset("waiting_for_silence")
        else:
            is_silence = self.is_silence(pcm)
            interrupted = self.interruption_detection.check_for_interruption(
                pcm, is_silence
            )
            if interrupted:
                logger.info("Interrupted")
                self.chat_gpt.restart()
                # Capture the last few frames when interrupting the assistent, drop anything before that, since we don't want any echo feedbacks
                self.recording_audio_buffer = self.recording_audio_buffer[
                    -frame_length * 32 * 2 :
                ]
                self.reset("waiting_for_silence")


def main():
    parser = argparse.ArgumentParser(
        description="BMO, the open-source voice assistant with replaceable parts"
    )
    parser.add_argument(
        "-sr",
        "--speech-recognition",
        dest="speech_recognition",
        choices=["whisper", "whisper-cpp"],
        default="whisper",
        help="Choose the speech recognition engine to be used, default to whisper",
    )
    parser.add_argument(
        "-tts",
        "--text-to-speech",
        dest="text_to_speech",
        choices=text_to_speech.ENGINES.keys(),
        default="native",
        help="Choose the text-to-speech engine to be used, default to native",
    )

    cli_args = parser.parse_args()

    start_time: Synchronized = Value("d", time.time())  # type: ignore
    log_formatter.start_time = start_time

    recorder = PvRecorder(device_index=-1, frame_length=frame_length)
    audio_recording = AudioRecording(recorder, cli_args)
    try:
        while True:
            audio_recording.next_frame()
    except KeyboardInterrupt:
        print("Stopping ...")
        audio_recording.stop()
    finally:
        recorder.delete()


if __name__ == "__main__":
    main()
