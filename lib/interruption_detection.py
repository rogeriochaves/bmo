from multiprocessing import Process, Queue
from queue import Empty
import librosa
from scipy.spatial.distance import cosine
from typing import Any, List
import numpy as np
import math

from lib.utils import terminate_pid_safely

# This code is quite convoluted, trying to do a hacky, but lightweight, way of detecting interruption on the fly.
# It is necessary because we cannot do a simple volume check, since there might be an echo loop when the speakers
# are too close from the microphone, and all previous attempts of echo cancellation I've tried, failed, it's very hard.
#
# The way this works is by getting both the pcm, that is the pcm data incoming from user's microphone,
# and the reply_audio, that is, the audio chunks produced by ElevenLabs. We capture the first 16 * 5 chunks of PCMs
# from both input and output, and we analyse their similarity using a very simple method of cosine similarity.
# If it's more than 96% similar (silence already achieves 94% similarity, so we need to be stricter), then we consider
# there is a feedback loop from the microphone
#
# Then, we behave in two different ways: if we decide there is no echo feedback loop, then it's easy, we try to detect
# a volume louder than the silence threshold, if that's the case, we interrupt the assistant audio. But in the case of
# having a feedback loop, looking at similarity only didn't work well enough, so we look at volume differences, that is
# if you speak during the silence that assistant has in between sentences, then we can detect a diff in volumes and use
# that as interruption
#
# For the first chunks similarity detection to work well, both the pcm and the reply_audio must be very aligned,
# but trial-and-error, I reached the guess number of 28 frames to skip after ffplay is started. Then, for the volume diff
# of the echo feedback loop scenario, we need even more strict match. For that, I detect the time the mic audio was first
# louder than the silence_threshold, and start counting the reply_audio from there, first trimming off the first silent part
# of it

silence_threshold = 300  # same as from main
frame_length = 512  # same as from main


class InterruptionDetection:
    reply_audio: List[int]
    reply_audio_skip_frames: int
    reply_audio_skip_frames_after_first_sound: int

    accumulated_similarity: List[Any]
    audio_playback_process_pid: int
    interrupted: bool = False
    done: bool = False
    interruption_check_process: Process
    interruption_check_in_queue: Queue
    interruption_check_out_queue: Queue
    total_checks_count: int

    def __init__(self, audio_playback_process_pid: int) -> None:
        self.reply_audio = []
        self.reply_audio_skip_frames = (
            frame_length * -28
        )  # this is the guessed delay between receiving the first buffer and ffplay starts playing it
        self.reply_audio_skip_frames_after_first_sound = (
            -1
        )  # this will start counting only when first sound is heard, for feedback loops
        self.accumulated_similarity = []
        self.audio_playback_process_pid = audio_playback_process_pid
        self.interruption_check_in_queue = Queue()
        self.interruption_check_out_queue = Queue()
        self.total_checks_count = 0

        self.interruption_check_process = Process(
            target=check_next_frame,
            args=(self.interruption_check_in_queue, self.interruption_check_out_queue),
        )
        self.interruption_check_process.start()

    def is_done(self):
        return self.done

    def stop(self):
        self.interruption_check_in_queue.put("done")
        self.done = True
        terminate_pid_safely(self.audio_playback_process_pid)
        self.interruption_check_process.kill()

    def check_for_interruption(self, pcm, is_silence: bool):
        if self.interrupted:
            return True

        if len(self.reply_audio) == 0:
            return False

        try:
            signal = self.interruption_check_out_queue.get(block=False)
            if signal == "interrupt":
                self.interrupted = True
                self.stop()
                return True
        except Empty:
            pass

        self.total_checks_count += 1

        if self.reply_audio_skip_frames_after_first_sound == -1 and not is_silence:
            self.reply_audio_skip_frames_after_first_sound = frame_length

            # trim silent beginning from reply_audio
            for i in range(0, len(self.reply_audio), frame_length):
                rms = np.sqrt(
                    np.mean(np.array(self.reply_audio[i : (i + frame_length)]) ** 2)
                )
                if rms < silence_threshold:
                    self.reply_audio_skip_frames_after_first_sound += frame_length
                else:
                    break

        self.reply_audio_skip_frames += frame_length
        if self.reply_audio_skip_frames_after_first_sound > -1:
            self.reply_audio_skip_frames_after_first_sound += frame_length

        if self.reply_audio_skip_frames < 0:
            return False

        if self.reply_audio_skip_frames >= len(self.reply_audio):
            self.stop()
            return False

        reply_audio_guessed_slice = self.reply_audio[
            self.reply_audio_skip_frames : self.reply_audio_skip_frames + frame_length
        ]
        reply_audio_after_first_sound_slice = self.reply_audio[
            self.reply_audio_skip_frames_after_first_sound : self.reply_audio_skip_frames_after_first_sound
            + frame_length
        ]

        self.interruption_check_in_queue.put(
            (pcm, reply_audio_guessed_slice, reply_audio_after_first_sound_slice)
        )
        return False


def check_next_frame(in_queue: Queue, out_queue: Queue):
    batch = []
    batch_size = 16
    how_many_initial_batchs_to_define = 5

    initial_batches_similarities = []
    has_audio_feedback = False

    stop_counts = 0
    loops_since_last_stop = 0

    while True:
        item = in_queue.get()
        if item == "done":
            break

        batch.append(item)
        if len(batch) < batch_size:
            continue

        pcm_batch = [y for x in batch for y in x[0]]
        prev_batch = batch
        batch = []

        if len(initial_batches_similarities) < how_many_initial_batchs_to_define:
            reply_batch = [y for x in prev_batch for y in x[1]]
            if len(reply_batch) < len(pcm_batch):
                pcm_batch = pcm_batch[: len(reply_batch)]
            similarity = calculate_similarity(pcm_batch, reply_batch)

            initial_batches_similarities.append(similarity)
            if len(initial_batches_similarities) == 5:
                if np.mean(initial_batches_similarities) > 0.95:
                    batch_size = 4
                    has_audio_feedback = True

            continue

        loops_since_last_stop += 1
        if (
            loops_since_last_stop == 4 * 2
        ):  # 4 * 2 (two times 16 frame batches) = ~4 seconds
            stop_counts = max(stop_counts - 1, 0)
            loops_since_last_stop = 0

        volume_pcm = np.sqrt(np.mean(np.array(pcm_batch) ** 2))

        if has_audio_feedback:
            reply_batch_after_first_sound = [y for x in prev_batch for y in x[2]]
            volume_reply = np.sqrt(
                np.mean(np.array(reply_batch_after_first_sound) ** 2)
            )
            silence_log_diff = math.log(volume_pcm or 1) - math.log(volume_reply or 1)

            if silence_log_diff > 3.5:
                stop_counts += 1
                loops_since_last_stop = 0
        else:
            if volume_pcm >= silence_threshold:
                stop_counts += 1
                loops_since_last_stop = 0

        if stop_counts >= 2:
            out_queue.put("interrupt")
            break


# Code from the interwebs
def calculate_similarity(y1, y2):
    y1 = np.array(y1).astype(float) / np.iinfo(np.int16).max
    y2 = np.array(y2).astype(float) / np.iinfo(np.int16).max

    # Extract MFCCs
    mfcc1 = librosa.feature.mfcc(y=y1, sr=16000)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=16000)

    # Calculate the mean of MFCCs
    mean_mfcc1 = np.mean(mfcc1.T, axis=0)
    mean_mfcc2 = np.mean(mfcc2.T, axis=0)

    # Compute cosine similarity between the means
    similarity = 1 - cosine(mean_mfcc1, mean_mfcc2)

    return similarity
