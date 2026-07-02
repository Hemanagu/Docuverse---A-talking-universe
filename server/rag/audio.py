"""
Audio Services: Speech-to-Text (STT) and Text-to-Speech (TTS)
============================================================
Uses free APIs:
- STT: SpeechRecognition using Google's free endpoint
- TTS: edge-tts (Microsoft Edge TTS API, high quality, free, no keys needed)
"""

import io
import os
import tempfile
import edge_tts
import speech_recognition as sr
from pydub import AudioSegment


async def text_to_speech(text: str, voice: str = "en-US-AriaNeural") -> bytes:
    """
    Generate speech audio (MP3) from text using edge-tts.
    Returns the mp3 bytes.
    """
    # Clean text
    clean_text = text.replace("📄", "").strip()
    if not clean_text:
        return b""

    communicate = edge_tts.Communicate(clean_text, voice)
    audio_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data += chunk["data"]

    return audio_data


def speech_to_text(audio_bytes: bytes, file_extension: str = "webm") -> str:
    """
    Convert uploaded audio bytes to text using SpeechRecognition (Google Free).
    Requires converting the webm/ogg file from browser to WAV using pydub.
    """
    recognizer = sr.Recognizer()

    # Create temporary files because pydub and SpeechRecognition work best with files
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_in:
        temp_in.write(audio_bytes)
        temp_in_path = temp_in.name

    temp_out_path = temp_in_path + ".wav"

    try:
        # Convert to WAV
        audio_segment = AudioSegment.from_file(temp_in_path)
        audio_segment.export(temp_out_path, format="wav")

        # Recognize
        with sr.AudioFile(temp_out_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except sr.UnknownValueError:
        return ""  # Could not understand audio
    except sr.RequestError as exc:
        print(f"[audio] Speech recognition API error: {exc}")
        return ""
    except Exception as exc:
        print(f"[audio] Processing error: {exc}")
        return ""
    finally:
        # Cleanup
        if os.path.exists(temp_in_path):
            os.remove(temp_in_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)
