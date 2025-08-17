# Special thanks for moral support adn ideation 
Supporter and collaborator : Abhay0603 

Github : https://github.com/abhay0603

Supporter and collaborator  : Vineet K Bharti

Github : https://github.com/vineet-phoenix
***

# Dub It Up

Dub It Up is a video translation tool that automatically dubs the audio in your input video into your chosen language. It also generates word-level transcripts with timing, making your video accessible for localization, subtitles, and educational purposes.

***

## Features

- **Automatic Video Translation**
  - Translate any spoken video to your desired language.
- **Word-Level Transcription**
  - Get a full transcript with word-by-word timing.
- **Authentic Dubbing**
  - Maintains the original emotion and pacing for natural dubbing results.

***

## Demo

> **Note:** Large video files in this repository are stored and managed using Git LFS due to GitHub’s file size restrictions.
>
> If you wish to download demo videos, please ensure you have Git LFS installed:
>
> ```bash
> git lfs install
> git lfs pull
> ```
>
> The sample video is not viewable as raw/download in-browser due to its size, but is available after cloning and pulling with LFS.

***

## Installation

```bash
git clone https://github.com/your-username/dub-it-up.git
cd dub-it-up
```

Install dependencies:
Dependecies are already setted in the code lines , justvrun them chronologically.

***

## Usage

Run video dubbing and transcription:

```bash
python dub_it_up.py --input input_video.mp4 --lang target_language
```
- `--input` : Path to your video file (must reside in the repository folder or be copied there).
- `--lang` : Output language code (e.g., `en` for English, `hi` for Hindi).

**Output:**
- Translated/dubbed video
- Word-level transcript (TXT/JSON)

***

## Example

```bash
python dub_it_up.py --input demo_video.mp4 --lang hi
```

***

## Output

- **Dubbed Video:** Video file with audio translated into your chosen language
- **Transcript:** Word-level transcript file with timestamps
- **Sample Files:** Samples included via Git LFS

***

## Contributing

Pull requests are welcome!  
For major changes, please open an issue first to discuss proposed modifications.

***

## License

Distributed under the MIT License.  
See `LICENSE` for more information.

***

## Acknowledgements

- [Gradio](https://gradio.app)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)
- [Edge-TTS](https://github.com/ranyelhousieny/edge-tts)
- [Transformers](https://github.com/huggingface/transformers)
- [Git Large File Storage](https://git-lfs.com)

***

Feel free to copy, customize, and add badges or links to your documentation, issues, or website. This will ensure your GitHub README looks professional, easy-to-read, and complete for users and contributors.
