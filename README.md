# Setup

Use python 3.7, because that's the latest one (as of time of writing) where numpy works out of the box on raspberry pi

Install dependencies:

```
pip install -r requirements.txt
```

Copy the `.env.sample` file to `.env` and fill the env vars. Get the OpenAI key on OpenAI's website, and Picovoice key from Picovoice Console (https://console.picovoice.ai/)

<!-- # TTS with Piper

Installing piper is quite tricky. On raspberry you can download the compiled version: https://github.com/rhasspy/piper

But on MacOS you need to install espeak-ng first, inside the lib folder from piper, but following the instructions [here](https://github.com/espeak-ng/espeak-ng/blob/master/docs/building.md), then install onnxruntime, then follow [this thread](https://github.com/rhasspy/piper/issues/27) -->
