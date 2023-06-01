# BMO Voice Assistant

![a picture of BMO from Adventure Time](./docs/BMO.webp)

BMO is a fast, open-source voice assistant using Speech Recognition (Whisper or whisper.cpp) + LLM (ChatGPT) + Text-To-Speech (espeak-ng, Elevenlabs or Piper) that runs on macOS and Raspberry PI with multi-language support

Here is a demo of it running on the Raspberry PI (unmmute the video):

https://github.com/rogeriochaves/bmo/assets/792201/6454ea77-b580-4dfa-9c3a-eabb847fc0ab

On the desktop it can run even faster:

https://github.com/rogeriochaves/bmo/assets/792201/56289d20-5be7-40da-afb6-cbbf22f0bcc6

## Features

✅ Wake up keyword detection

✅ Interruption detection

✅ Streamed speech recognition

✅ Streamed chatgpt reply

✅ Streamed text-to-speech audio

✅ Multi-language (basically it understands all languages with Whisper and ChatGPT, and speaks English, German, Polish, Spanish, Italian, French, Portuguese and Hindi with Elevenlabs)

## Installation

To run BMO, first you will need to have at least python 3.7 version installed on your macOS or Raspberry PI, check it with:

```
python -V
```

Then, clone this repo and install run the install script to download the dependencies:

```
git clone git@github.com:rogeriochaves/bmo.git
cd bmo
./install.sh
```

Now, create a `.env` file in the bmo folder to put your API keys. You will need at the very least the `OPENAI_API_KEY`, which you can get on [OpenAI website](https://platform.openai.com/account/api-keys), to be able to use ChatGPT API:

```
OPENAI_API_KEY=<required from openai.com>
PICOVOICE_ACCESS_KEY=<optional from picovoice.ai>
ELEVEN_LABS_API_KEY=<optional from elevenlabs.io>
```

Finally, launch BMO and start talking to it!

```
python main.py
```

You can also pass --help for more options:

```
python main.py --help
```

## Text to Speech Engine

By default, native tts is used, which is the `say` command on the mac, or the `espeak-ng` on raspberry pi. Those are very robotic and poor quality voices, but also realtime for a good speaking experience.

On the mac, you can improve the quality of the `say` tts immediately, by simply going to System Settings > Accessibility > Spoken Content and choosing Siri voice in System voice, so you will have as high quality voice as Siri. If you click "Manage Voices..." you can download more voices.

However, if you really want State of the Art Text-To-Speech, with multilanguage output with native speaker quality, Elevenlabs has the best model today. To use it, grab your API key with them and fill the `ELEVEN_LABS_API_KEY` field on `.env`. Then, start BMO with elevenlabs as tts:

```
python main.py -tts elevenlabs
```

On the Raspberry Pi, if you want to use something that is faster than Elevenlabs, but with as high quality as Siri, then you can user Piper, but only if your Raspberry Pi was installed with the 64 bit version, which should be the case for the newer installations. To use piper, first run the `piper_install.sh` script on your Raspberry Pi:

```
./piper_install.sh
```

Then, start BMO with Piper as TTS:

```
python main.py -tts piper
```

## Speech Recognition Engines

By default, Whisper API from OpenAI is used for speech recognition, it is super fast and understands all languages, but you can also use [whisper.cpp](https://github.com/ggerganov/whisper.cpp) instead, an optimized version to run locally on all platforms.

To use whisper.cpp instead of Whisper API, first clone whisper.cpp repo inside bmo folder:

```
cd bmo
git clone https://github.com/ggerganov/whisper.cpp
```

Then download the model (this one only understands english), and build whisper.cpp:

```
cd whisper.cpp
./models/download-ggml-model.sh medium.en
make
```

Now run BMO with whisper.cpp:

```
python main.py -sr whisper-cpp
```

Here is a demo of the multi-language capability while keeping the same voice of Elevenlabs

https://github.com/rogeriochaves/bmo/assets/792201/5e7fdadd-1751-476e-9ff8-ff459ba9834c

## Standby Mode and Wake Up Word Detection

If you are going to run the assistant for longer, then you probably want to enable a wake up word, otherwise all the audio captured by the microphone will keep being streamed to the Text to Speech engine for transcription, additionally, if you leave it running on the Raspberry Pi, it will waste a lot of CPU. So instead you can enable the wake up word detection to have a behaviour similar to Alexa or Google Assistant.

By default BMO is configured to listen to the keyword **Chat G-P-T** (with english pronunciation, saying G-P-T letter by letter), which is detected by using very efficient and accurate processing by the [porcupine](https://github.com/Picovoice/porcupine) library. On my personal tests, I had no false positives ever, and the CPU from Raspberry Pi stays around 8% usage, which, with my passive cooling case, keeps the temperature below 40ºC, so I can leave it running the whole day.

To set it up, go to [picovoice.ai](https://picovoice.ai) and get an API key, registering for a free account is enough, and then put it on `PICOVOICE_ACCESS_KEY`. You can change the keyword to be detected on the `porcupine.py` file.

BMO will enter in standby automatically after 20s of silence, or if you say bye to it, in any way or form, ChatGPT understands that you are saying bye.

## Initial Prompt and Personality

BMO has an initial prompt to have a very friendly personality, speaking a lot of slangs, and giving very short replies, so it is better for keeping a casual conversation. Feel free to change the prompt and play with it's personality, the initial prompt is in the `lib/chatgpt.py` file, change it there to see the effects.
