# Fichier : app/core/tts_model_def.py
import io
import os
import re

import numpy as np
import threading
from typing import Optional
import soundfile as sf
from kokoro import KPipeline
import numpy as np

_pipelines = {}
_pipelines_lock = threading.Lock()

_kokoro_pipelines = {}


def clean_for_speech(text: str) -> str:
    """Retire ce qui se lit mal à voix haute (Markdown, Mermaid, liens)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # blocs de code / Mermaid
    text = re.sub(r"https?://\S+", " ", text)  # liens
    text = re.sub(r"^\s*Lien\s*:\s*$", " ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*(\d+)\.\s*", r"Étape \1. ", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-•*]\s+", "", text, flags=re.MULTILINE)  # puces
    text = re.sub(r"[*_#`>«»]", "", text)
    text = re.sub(r"\n{2,}", ". ", text)
    return re.sub(r"\s+", " ", text).strip()


def _get_kokoro(language: str):
    from kokoro import (
        KPipeline,
    )  # import tardif : requis seulement si TTS_ENGINE=kokoro

    code = "f" if language.lower().startswith("fr") else "a"
    if code not in _kokoro_pipelines:
        with _pipelines_lock:
            if code not in _kokoro_pipelines:
                _kokoro_pipelines[code] = KPipeline(lang_code=code)
    return _kokoro_pipelines[code]


def _kokoro_speech_bytes(text: str, language: str = "fr") -> bytes:
    pipeline = _get_kokoro(language)
    voice = "ff_siwis" if language.lower().startswith("fr") else "af_heart"
    parts = []
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        if audio is not None:
            parts.append(
                audio.detach().cpu().numpy()
                if hasattr(audio, "detach")
                else np.asarray(audio)
            )
    if not parts:
        return b""
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(parts), 24000, format="WAV")
    return buffer.getvalue()


def get_tts_pipeline(lang_code: str) -> KPipeline:
    """Initialise et met en cache le pipeline Kokoro selon le code langue ('f' ou 'a')."""
    global _pipelines
    if lang_code not in _pipelines:
        with _pipelines_lock:
            if lang_code not in _pipelines:
                print(
                    f"Chargement du modèle TTS Kokoro pour la langue : {lang_code}..."
                )
                _pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _pipelines[lang_code]


def generate_speech_bytes(text: str, language: str = "fr") -> bytes:
    """Génère un flux audio WAV adapté à la langue détectée ('fr' ou 'en')."""
    try:
        text = clean_for_speech(text)
        if not text:
            return b""
        if os.getenv("TTS_ENGINE", "melo").lower() == "kokoro":
            return _kokoro_speech_bytes(text, language)
        # Configuration de la voix selon la langue
        if language.lower().startswith("en"):
            lang_code = "a"
            voice = "af_bella"  # Voix anglaise claire
        else:
            lang_code = "f"
            voice = "ff_siwis"  # Voix française claire

        pipeline = get_tts_pipeline(lang_code)
        generator = pipeline(text, voice=voice, speed=1.0)

        audio_segments = []
        for _, _, audio in generator:
            audio_segments.append(audio)

        if not audio_segments:
            return b""

        full_audio = np.concatenate(audio_segments)

        buffer = io.BytesIO()
        sf.write(buffer, full_audio, 24000, format="WAV")
        return buffer.getvalue()
    except Exception as e:
        print(f"[TTS Error] Échec de la synthèse vocale : {e}")
        return b""
