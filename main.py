from dotenv import load_dotenv  # has to be the first import

load_dotenv()
from lib.delta_logging import logging, red, reset  # has to be the second
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, List, Optional
from typing_extensions import Literal
from lib.interruption_detection import InterruptionDetection
from lib.porcupine import wakeup_keywords
from lib.utils import calculate_volume
import lib.chatgpt as chatgpt
from lib.chatgpt import Conversation, Message, initial_message
import lib.elevenlabs as elevenlabs
from lib.whisper import WhisperTranscriber
import os
import struct
import pvporcupine
from pvrecorder import PvRecorder
import openai

picovoice_access_key = os.environ["PICOVOICE_ACCESS_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

logger = logging.getLogger()

porcupine = pvporcupine.create(
    access_key=picovoice_access_key,
    keyword_paths=wakeup_keywords(),
)
frame_length = porcupine.frame_length  # 512
buffer_size_when_not_listening = frame_length * 32 * 20  # keeps 20s of audio
buffer_size_on_active_listening = frame_length * 32 * 60  # keeps 60s of audio
sample_rate = 16000  # sample rate for Porcupine is fixed at 16kHz
silence_threshold = 300  # maybe need to be adjusted
# TODO: reduce to 1s and stop if needed
silence_limit = 0.5 * sample_rate // frame_length  # 0.5 seconds of silence
speaking_minimum = 0.3 * sample_rate // frame_length  # 0.3 seconds of speaking
silence_time_to_standby = (
    10 * sample_rate // frame_length
)  # goes back to wakeup word checking after 10s of silence


RecordingState = Literal[
    "waiting_for_wakeup",
    "waiting_for_silence",
    "waiting_for_next_frame",
    "start_reply",
    "replying",
]


class AudioRecording:
    state: RecordingState
    silence_frame_count: int
    speaking_frame_count: int
    recording_audio_buffer: bytearray
    recorder: PvRecorder
    reply_process: Optional[Process] = None
    reply_out_queue: Queue
    interruption_detection: Optional[InterruptionDetection] = None
    conversation: Conversation = [initial_message]
    transcriber: WhisperTranscriber

    def __init__(self, recorder: PvRecorder) -> None:
        self.recorder = recorder
        self.reply_out_queue = Queue()
        self.recording_audio_buffer = bytearray()
        self.speaking_frame_count = 0
        self.transcriber = WhisperTranscriber()
        self.reset("waiting_for_silence")

    def reset(self, state):
        self.recorder.start()
        self.state = state

        self.silence_frame_count = 0

        if state == "waiting_for_silence":
            self.transcriber.restart()
            self.interruption_detection = None
        elif state == "replying":
            self.interruption_detection = InterruptionDetection()
        else:
            self.speaking_frame_count = 0
            self.recording_audio_buffer = bytearray()
            self.interruption_detection = None

    def stop(self):
        self.recorder.stop()
        if self.reply_process:
            self.reply_process.kill()
        if self.interruption_detection:
            self.interruption_detection.stop()
        self.transcriber.stop()

    def next_frame(self):
        pcm = self.recorder.read()

        self.recording_audio_buffer.extend(struct.pack("h" * len(pcm), *pcm))
        self.drop_early_recording_audio_frames()

        if self.state == "waiting_for_wakeup":
            self.waiting_for_wakeup(pcm)

        elif self.state == "waiting_for_silence":
            self.waiting_for_silence(pcm)

        elif self.state == "waiting_for_next_frame":
            self.state = "start_reply"

        elif self.state == "start_reply":
            if self.reply_process is not None:
                self.reply_process.kill()  # kill previous process to not have one reply on top of the other
            self.recorder.stop()

            transcription = self.transcriber.transcribe_and_stop()

            self.reply_out_queue = Queue()
            self.reply_process = Process(
                target=reply,
                args=(
                    self.conversation,
                    transcription,
                    self.reply_out_queue,
                ),
            )
            self.reply_process.start()

            self.reset("replying")
            self.recorder.start()

        elif self.state == "replying":
            self.replying_loop(pcm)

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
        print(f"⚪️ Waiting for wake up word...", end="\r", flush=True)
        trigger = porcupine.process(pcm)
        if trigger >= 0:
            logger.info("Detected wakeup word #%s", trigger)
            elevenlabs.play_audio_file_non_blocking("beep2.mp3")
            self.state = "waiting_for_next_frame"

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
                    -frame_length:
                ]
            self.speaking_frame_count += 1
            self.silence_frame_count = 0

        if (
            self.silence_frame_count >= silence_limit
            and self.speaking_frame_count >= speaking_minimum
        ):
            logger.info("Detected silence a while after speaking, giving a reply")
            self.state = "waiting_for_next_frame"

        if self.silence_frame_count >= silence_time_to_standby:
            logger.info("Long silence time, going back to waiting for the wakeup word")
            self.silence_frame_count = 0
            self.speaking_frame_count = 0
            self.state = "waiting_for_wakeup"

    def replying_loop(self, pcm: List[Any]):
        if self.interruption_detection is None:
            return

        try:
            (action, data) = self.reply_out_queue.get(block=False)
            if action == "user_message":
                self.recording_audio_buffer = self.recording_audio_buffer[
                    -frame_length:
                ]
                self.conversation.append(data)
            elif action == "assistent_message":
                self.conversation.append(data)
            elif action == "play_beep":
                self.interruption_detection.pause_for(32)
                elevenlabs.play_audio_file_non_blocking(data)
            elif action == "reply_audio_started":
                self.silence_frame_count = 0
                self.speaking_frame_count = 0
                self.interruption_detection.start_reply_interruption_check(data)
            elif action == "reply_audio_ended":
                logger.info("Playing audio done")
                self.interruption_detection.stop()
        except Empty:
            pass

        if self.interruption_detection.is_done():
            self.recording_audio_buffer = bytearray()
            self.reset("waiting_for_silence")
        else:
            is_silence = self.is_silence(pcm)
            interrupted = self.interruption_detection.check_for_interruption(
                pcm, is_silence
            )
            if interrupted:
                logger.info("Interrupted")
                if self.reply_process:
                    self.reply_process.kill()
                # Capture the last few frames when interrupting the assistent, drop anything before that, since we don't want any echo feedbacks
                self.recording_audio_buffer = self.recording_audio_buffer[
                    -frame_length * 5 :
                ]
                self.reset("waiting_for_silence")


def reply(conversation: Conversation, transcription: str, reply_out_queue: Queue):
    try:
        logger.info("Transcription: %s", transcription)
        if len(transcription.strip()) > 1:
            user_message: Message = {"role": "user", "content": transcription}
            conversation.append(user_message)
            reply_out_queue.put(("user_message", user_message))
        else:
            logger.info("Transcription too small, probably a mistake, bailing out")
            reply_out_queue.put(("reply_audio_ended", None))
            if conversation[-1]["role"] != "user":
                return

        reply_out_queue.put(("play_beep", "beep.mp3"))

        reply = chatgpt.reply(conversation, reply_out_queue)

        logger.info("Reply: %s", reply["content"])
    except Exception:
        logging.exception("Exception thrown in reply")
        elevenlabs.play_audio_file("error.mp3", reply_out_queue)


def main():
    recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)

    audio_recording = AudioRecording(recorder)
    try:
        while True:
            audio_recording.next_frame()
    except KeyboardInterrupt:
        print("Stopping ...")
        audio_recording.stop()
    finally:
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()
