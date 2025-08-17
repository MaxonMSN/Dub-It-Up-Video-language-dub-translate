Dub It Up
Dub It Up is a powerful video translation tool that automatically translates voice in any video into your chosen language, generating a complete transcript and word-level timing for every utterance. This makes dubbing, localization, and video accessibility effortless for creators, educators, and media professionals.

Table of Contents
Features

Video Requirements

Installation

Usage

Demo

Output

Contributing

License

Features
Automatic Video Translation:
Translate any spoken video into your target language (supports multiple languages).

Word-Level Transcript:
Generates a transcript with precise word timings.

Authentic Dub:
Maintains original emotional tone and pacing for natural dubbing.

Large Video Support:
Handles large video files using Git Large File Storage (LFS).

Video Requirements
Input videos should be in .mp4 format (other formats supported with ffmpeg).

For videos larger than 100MB, this repository uses Git LFS to store and manage files. If you clone the repo and want the sample video, make sure you have Git LFS installed.

Installation
Clone this repository (with LFS):

bash
git clone https://github.com/your-username/dub-it-up.git
git lfs pull
Install dependencies:

bash
pip install -r requirements.txt
Required libraries: gradio, requests, edge-tts, pydub, faster-whisper, transformers, sentencepiece, torch, torchaudio (see requirements.txt).

Usage
Run the dubbing program:

bash
python dub_it_up.py --input input_video.mp4 --lang target_language
Replace input_video.mp4 with your video file and target_language (e.g., hi for Hindi, es for Spanish).

Output:

Translated video file (dubbed in your chosen language).

Word-level transcript (TXT or JSON).

Demo
Due to file size limits, the sample video is stored with Git LFS.
You may not be able to preview it directly on GitHub, but you can download and run it locally after cloning.

Output
Dubbed Video: Video file in the target language.

Transcript: Word-wise transcript with timings.

Examples: Sample input/output files provided (via Git LFS).

Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

License
Distributed under the MIT License. See LICENSE for more information.

Created by [Your Name/Handle], 2025
