import multiprocessing
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, List, Optional, Union
import numpy as np

from lib.utils import terminate_pid_safely, calculate_volume

# This code is a lightweight way of detecting interruption on the fly.
#
# The way it works is by trying to match when the audio from the assistant reply started playing, and measure the
# average volume on the first 16 * 5 chunks of PCMs, if there is a feedback loop from speaker to mic, the average volume
# will be higher than if there is silence
#
# So, at every frame, we detect if the mic volume suddenly got louder than the initial 5 batch frames average, if so, we
# interrupt the assistant
#
# Additionally, if the assistant has not started speaking yet, we do simple above silence_threshold interruption detection

silence_threshold = 300  # same as from main
frame_length = 512  # same as from main
pre_interrupt_speaking_minimum = (
    0.1 * 32
)  # 0.1 seconds of speaking to interrupt before audio_playback is reproduced


class InterruptionDetection:
    reply_audio_started: bool
    accumulated_similarity: List[Any]
    audio_playback_process_pid: Optional[int]
    interrupted: bool
    done: bool
    interruption_check_process: Process
    interruption_check_in_queue: Queue
    interruption_check_out_queue: Queue
    speaking_frame_count: int
    pause_frame_count: int

    def __init__(self) -> None:
        self.start()

    def start(self):
        self.reset()
        self.interruption_check_in_queue = multiprocessing.Queue()
        self.interruption_check_out_queue = multiprocessing.Queue()
        self.speaking_frame_count = 0
        self.pause_frame_count = 0

        self.interruption_check_process = Process(
            target=check_next_frame,
            args=(self.interruption_check_in_queue, self.interruption_check_out_queue),
        )
        self.interruption_check_process.start()

    def reset(self):
        self.reply_audio_started = False
        self.interrupted = False
        self.done = False
        self.audio_playback_process_pid = None

    def stop(self):
        self.interruption_check_in_queue.close()
        self.interruption_check_out_queue.close()
        self.done = True
        terminate_pid_safely(self.audio_playback_process_pid)
        self.interruption_check_process.kill()

    def pause_for(self, n_frames: int):
        self.pause_frame_count = n_frames

    def is_done(self):
        return self.done

    def start_reply_interruption_check(self, audio_playback_process_id: int):
        self.audio_playback_process_pid = audio_playback_process_id
        self.reply_audio_started = True

    def interrupt(self):
        self.interrupted = True

    def should_stop_consuming_microphone(self):
        return self.reply_audio_started

    def check_for_interruption(self, pcm: Union[List[Any], bytearray], is_silence: bool):
        if self.interrupted:
            return True

        if self.pause_frame_count > 0:
            self.pause_frame_count -= 1
            return False

        if not self.reply_audio_started:
            if is_silence:
                return False
            else:
                self.speaking_frame_count += 1
                if self.speaking_frame_count > pre_interrupt_speaking_minimum:
                    self.interrupt()
                    return True
        return False

        try:
            signal = self.interruption_check_out_queue.get(block=False)
            if signal == "interrupt":
                self.interrupt()
                return True
        except Empty:
            pass

        self.interruption_check_in_queue.put(pcm)
        return False


def check_next_frame(in_queue: Queue, out_queue: Queue):
    batch = []
    batch_size = 16
    how_many_initial_batchs_to_define = 5

    initial_batches_volume = []
    max_volume = 0

    stop_counts = 0
    loops_since_last_stop = 0

    while True:
        item = in_queue.get()

        batch.append(item)
        if len(batch) < batch_size:
            continue

        pcm_batch = [x for x in batch]
        batch = []

        if len(initial_batches_volume) < how_many_initial_batchs_to_define:
            initial_batches_volume.append(calculate_volume(pcm_batch))
            if len(initial_batches_volume) == how_many_initial_batchs_to_define:
                max_volume = max(
                    float(np.quantile(initial_batches_volume, 0.9)), silence_threshold
                )
                batch_size = 4

            continue

        loops_since_last_stop += 1
        if (
            loops_since_last_stop == 4 * 2
        ):  # 4 * 2 (two times 16 frame batches) = ~4 seconds
            stop_counts = max(stop_counts - 1, 0)
            loops_since_last_stop = 0

        volume_pcm = calculate_volume(pcm_batch)

        if volume_pcm >= max_volume:
            stop_counts += 1
            loops_since_last_stop = 0

        if stop_counts >= 2:
            out_queue.put("interrupt")
            break
