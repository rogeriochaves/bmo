from io import BytesIO
import subprocess
import wave


def transcribe(file) -> str:
    whispercpp = subprocess.Popen(
        args=[
            "./whisper.cpp/main",
            "-m",
            "./whisper.cpp/models/ggml-tiny.en.bin",
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


class WhisperTranscriber:
    whispercpp: subprocess.Popen[bytes]
    first: bool

    def __init__(self) -> None:
        self.start()

    def start(self):
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
                "5000"
                # "-nt",
                # "-f",
                # "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.first = True

    def consume(self, audio_buffer):
        pass

    def create_audio_file(self, audio_buffer):
        virtual_file = BytesIO()
        wav_file = wave.open(virtual_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(audio_buffer)
        wav_file.close()
        virtual_file.name = "recording.wav"
        virtual_file.seek(0)

        return virtual_file

    def stop(self):
        self.whispercpp.stdin.close()  # type: ignore

    def transcribe_and_stop(self):
        self.whispercpp.terminate()
        output, _ = self.whispercpp.communicate()
        output_lines = output.decode().split("\n")
        output_lines = [line.split("\x1b[2K\r")[-1].strip() for line in output_lines]
        output = "\n".join([line for line in output_lines if line != ""])

        return output
