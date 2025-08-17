
from google.colab import drive, output
!huggingface-cli login

# Core libraries
!pip install gradio         # For building the UI
!pip install requests       # For HTTP requests

# Text-to-speech (edge_tts)
!pip install edge-tts nest_asyncio     # Microsoft Edge TTS service

# Audio processing
!pip install pydub          # Audio manipulation (requires ffmpeg)
!sudo apt-get install ffmpeg # For Linux (if not already installed)
# For Windows: Download ffmpeg from https://ffmpeg.org/download.html

# ASR (Whisper)
!pip install faster-whisper # Faster Whisper implementation
!pip install numpy          # Required for audio array handling

# NLP (Transformers)
!pip install transformers   # Hugging Face Transformers library
!pip install sentencepiece  # Often required for Transformers models

# Optional but recommended for GPU users
!pip install torch torchaudio --extra-index-url https://download.pytorch.org/whisper/stables.html

import gradio as gr
import requests
import edge_tts
import asyncio
from pydub import AudioSegment
import tempfile
import os
import numpy as np
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Whisper model
model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cpu",
    compute_type="int8"
)

# Initialize NLLB model
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Language and voice configuration
LANGUAGE_MAPPING = {
    "English": {
        "code": "eng_Latn",
        "Male": "en-US-ChristopherNeural",
        "Female": "en-US-JennyNeural"
    },
    "Hindi": {
        "code": "hin_Deva",
        "Male": "hi-IN-MadhurNeural",
        "Female": "hi-IN-SwaraNeural"
    },
    "French": {
        "code": "fra_Latn",
        "Male": "fr-FR-HenriNeural",
        "Female": "fr-FR-DeniseNeural"
    },
    "Spanish": {
        "code": "spa_Latn",
        "Male": "es-ES-AlvaroNeural",
        "Female": "es-ES-ElviraNeural"
    },
    "Arabic": {
        "code": "arb_Arab",
        "Male": "ar-SA-HamedNeural",
        "Female": "ar-SA-ZariyahNeural"
    },
    "German": {
        "code": "deu_Latn",
        "Male": "de-DE-ConradNeural",
        "Female": "de-DE-KatjaNeural"
    },
    "Italian": {
        "code": "ita_Latn",
        "Male": "it-IT-DiegoNeural",
        "Female": "it-IT-ElsaNeural"
    },
    "Japanese": {
        "code": "jpn_Jpan",
        "Male": "ja-JP-KeitaNeural",
        "Female": "ja-JP-NanamiNeural"
    }
}

# Gender detection API
GENDER_API_URL = "https://api-inference.huggingface.co/models/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
HEADERS = {"Authorization": "Bearer your_hf_token"}

def detect_gender(audio_chunk_path):
    """Detect gender from audio chunk using API"""
    try:
        with open(audio_chunk_path, "rb") as f:
            data = f.read()
        response = requests.post(GENDER_API_URL, headers=HEADERS, data=data)
        result = response.json()
        return result[0]['label'].capitalize() if isinstance(result, list) else "Unknown"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_audio_chunk(full_audio_path, start_time, end_time):
    """Extract precise audio segment"""
    audio = AudioSegment.from_file(full_audio_path)
    chunk = audio[max(0, start_time*1000):end_time*1000]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        chunk.export(temp_file.name, format="wav")
        return temp_file.name

def translate_text(text, target_lang):
    """Translate text using NLLB model"""
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    translated_tokens = nllb_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANGUAGE_MAPPING[target_lang]["code"]),
        max_length=512
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

async def generate_tts(text: str, voice: str, available_time: float) -> AudioSegment:
    """Generate TTS with speed adjustment"""
    communicate = edge_tts.Communicate(text, voice)
    temp_path = 'temp.mp3'
    await communicate.save(temp_path)

    audio = AudioSegment.from_mp3(temp_path)
    natural_duration = len(audio) / 1000

    # Adjust speed if needed
    rate = "+0%"
    if natural_duration > available_time > 0:
        speed_ratio = min(natural_duration / available_time, 1.5)
        rate = f"+{int((speed_ratio - 1) * 100)}%"

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(temp_path)
    adjusted_audio = AudioSegment.from_mp3(temp_path)
    os.remove(temp_path)
    return adjusted_audio

async def synthesize_dubbed_audio(segments, target_lang):
    """Generate multilingual dubbed audio with gender-specific voices"""
    if not segments or target_lang not in LANGUAGE_MAPPING:
        return None

    total_duration = max(seg["end"] for seg in segments)
    final_audio = AudioSegment.silent(duration=int(total_duration * 1000))

    for seg in segments:
        target_lang_info = LANGUAGE_MAPPING[target_lang]
        voice = target_lang_info.get(seg["gender"], target_lang_info["Female"])

        translated_text = translate_text(seg["text"], target_lang)
        available_time = seg["end"] - seg["start"]

        audio_segment = await generate_tts(translated_text, voice, available_time)

        target_ms = int(available_time * 1000)
        if len(audio_segment) > target_ms:
            audio_segment = audio_segment[:target_ms]
        else:
            audio_segment += AudioSegment.silent(duration=target_ms - len(audio_segment))

        position = int(seg["start"] * 1000)
        final_audio = final_audio.overlay(audio_segment, position=position)

    final_output = "dubbed_output.mp3"
    final_audio.export(final_output, format="mp3")
    return final_output

def transcribe_audio(audio_path):
    """Transcribe audio with word-level timestamps"""
    segments, info = model.transcribe(audio_path, beam_size=3, word_timestamps=True)

    transcript = f"Detected language: {info.language}\n\n"
    segment_data = []

    for segment in segments:
        if not segment.words:
            continue

        word_start = segment.words[0].start
        word_end = segment.words[-1].end

        chunk_path = extract_audio_chunk(audio_path, word_start, word_end)
        gender = detect_gender(chunk_path)

        transcript += f"[{gender}] [{word_start:.2f}s → {word_end:.2f}s] {segment.text}\n"
        segment_data.append({
            "text": segment.text,
            "gender": gender,
            "start": word_start,
            "end": word_end
        })

    return transcript, segment_data

with gr.Blocks() as demo:
    gr.Markdown("# Multilingual Dubbing System")

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath")
        lang_dropdown = gr.Dropdown(list(LANGUAGE_MAPPING.keys()), label="Target Language", value="Hindi")

    with gr.Row():
        transcript_output = gr.Textbox(label="Original Transcript", lines=10)
        tts_output = gr.Audio(label="Dubbed Audio", interactive=False)

    with gr.Row():
        transcribe_btn = gr.Button("Transcribe Audio")
        dub_btn = gr.Button("Generate Dubbed Version")

    state = gr.State()

    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=[transcript_output, state]
    )

    dub_btn.click(
        fn=synthesize_dubbed_audio,
        inputs=[state, lang_dropdown],
        outputs=tts_output
    )

demo.launch(share=True)

import gradio as gr
import requests
import edge_tts
import asyncio
from pydub import AudioSegment
import tempfile
import os
import subprocess
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize models
whisper_model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cpu",
    compute_type="int8"
)

nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Language and voice configuration
LANGUAGE_MAPPING = {
    "English": {
        "code": "eng_Latn",
        "Male": "en-US-ChristopherNeural",
        "Female": "en-US-JennyNeural"
    },
    "Hindi": {
        "code": "hin_Deva",
        "Male": "hi-IN-MadhurNeural",
        "Female": "hi-IN-SwaraNeural"
    },
    "French": {
        "code": "fra_Latn",
        "Male": "fr-FR-HenriNeural",
        "Female": "fr-FR-DeniseNeural"
    },
    "Spanish": {
        "code": "spa_Latn",
        "Male": "es-ES-AlvaroNeural",
        "Female": "es-ES-ElviraNeural"
    },
    "Arabic": {
        "code": "arb_Arab",
        "Male": "ar-SA-HamedNeural",
        "Female": "ar-SA-ZariyahNeural"
    },
    "German": {
        "code": "deu_Latn",
        "Male": "de-DE-ConradNeural",
        "Female": "de-DE-KatjaNeural"
    },
    "Italian": {
        "code": "ita_Latn",
        "Male": "it-IT-DiegoNeural",
        "Female": "it-IT-ElsaNeural"
    },
    "Japanese": {
        "code": "jpn_Jpan",
        "Male": "ja-JP-KeitaNeural",
        "Female": "ja-JP-NanamiNeural"
    }
}

# Gender detection API
GENDER_API_URL = "https://api-inference.huggingface.co/models/alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
HEADERS = {"Authorization": "Bearer your_hf_token"}

def extract_audio(video_path):
    """Extract audio from video using FFmpeg"""
    output_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        output_audio
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return output_audio

def merge_audio_with_video(video_path, audio_path, output_path):
    """Merge new audio with original video"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
    return output_path

async def generate_tts(text: str, voice: str, available_time: float) -> AudioSegment:
    """Generate TTS with speed adjustment"""
    communicate = edge_tts.Communicate(text, voice)
    temp_path = 'temp.mp3'
    await communicate.save(temp_path)

    audio = AudioSegment.from_mp3(temp_path)
    natural_duration = len(audio) / 1000

    rate = "+0%"
    if natural_duration > available_time > 0:
        speed_ratio = min(natural_duration / available_time, 1.5)
        rate = f"+{int((speed_ratio - 1) * 100)}%"

    communicate = edge_tts.Communicate(text, voice, rate=rate)
    await communicate.save(temp_path)
    adjusted_audio = AudioSegment.from_mp3(temp_path)
    os.remove(temp_path)
    return adjusted_audio

async def synthesize_dubbed_audio(segments, target_lang, original_video_path):
    """Generate final dubbed video"""
    if not segments or target_lang not in LANGUAGE_MAPPING:
        return None

    # Create silent audio track
    total_duration = max(seg["end"] for seg in segments)
    final_audio = AudioSegment.silent(duration=int(total_duration * 1000))

    # Generate dubbed audio
    for seg in segments:
        lang_info = LANGUAGE_MAPPING[target_lang]
        voice = lang_info.get(seg["gender"], lang_info["Female"])
        translated_text = translate_text(seg["text"], target_lang)

        audio_segment = await generate_tts(
            translated_text,
            voice,
            seg["end"] - seg["start"]
        )

        target_ms = int((seg["end"] - seg["start"]) * 1000)
        if len(audio_segment) > target_ms:
            audio_segment = audio_segment[:target_ms]
        else:
            audio_segment += AudioSegment.silent(duration=target_ms - len(audio_segment))

        position = int(seg["start"] * 1000)
        final_audio = final_audio.overlay(audio_segment, position=position)

    # Save temp audio and merge with video
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    final_audio.export(temp_audio, format="wav")

    output_video = tempfile.NamedTemporaryFile(suffix="_dubbed.mp4", delete=False).name
    return merge_audio_with_video(original_video_path, temp_audio, output_video)

def transcribe_and_process(video_path):
    """Full processing pipeline"""
    # Extract audio
    audio_path = extract_audio(video_path)

    # Transcribe
    segments, info = whisper_model.transcribe(audio_path, beam_size=3, word_timestamps=True)

    transcript = f"Detected language: {info.language}\n\n"
    segment_data = []

    for segment in segments:
        if not segment.words:
            continue

        word_start = segment.words[0].start
        word_end = segment.words[-1].end

        chunk_path = extract_audio_chunk(audio_path, word_start, word_end)
        gender = detect_gender(chunk_path)

        transcript += f"[{gender}] [{word_start:.2f}s → {word_end:.2f}s] {segment.text}\n"
        segment_data.append({
            "text": segment.text,
            "gender": gender,
            "start": word_start,
            "end": word_end
        })

    return transcript, segment_data, video_path

with gr.Blocks() as demo:
    gr.Markdown("# DUB IT UP")

    with gr.Row():
        video_input = gr.Video(label="Upload Video", sources=["upload"])
        lang_dropdown = gr.Dropdown(list(LANGUAGE_MAPPING.keys()), label="Target Language", value="Hindi")

    with gr.Row():
        transcript_output = gr.Textbox(label="Original Transcript", lines=10)
        video_output = gr.Video(label="Dubbed Video")

    with gr.Row():
        process_btn = gr.Button("Process Video")

    state = gr.State()

    process_btn.click(
        fn=transcribe_and_process,
        inputs=video_input,
        outputs=[transcript_output, state, video_input]
    ).then(
        fn=synthesize_dubbed_audio,
        inputs=[state, lang_dropdown, video_input],
        outputs=video_output
    )

demo.launch(share=True)