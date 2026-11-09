# Fichier : app/core/audio_model_def.py
"""
import os
import tempfile

import httpx
import torch
from huggingface_hub import login
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

AUDIO_PROVIDER = os.getenv("AUDIO_PROVIDER", "hf").lower()


def initialize_stt_model():
    # Initialise et retourne le pipeline Speech-to-Text (ex: Whisper) selon le provider (HF ou autre).
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))

    model_id = "openai/whisper-small"  # Modèle léger et performant pour la transcription locale
    print(f"Chargement du modèle STT Hugging Face ({model_id})...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_id)

    stt_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    print("Modèle STT chargé avec succès !")
    return stt_pipe


# Instance globale partagée
stt_pipeline = initialize_stt_model()


async def transcribe_audio_file(file_path: str) -> str:
    # transcrit un fichier audio en texte brut en utilisant le pipeline STT configuré.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {file_path}")
    try:
        result = stt_pipeline(file_path)
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[STT Error] {e}")
        return ""
"""

# Fichier : app/core/audio_model_def.py
"""
Speech-to-Text avec faster-whisper (CTranslate2).

Dépendance : pip install faster-whisper
(plus besoin de torch, transformers, huggingface_hub, httpx)

Variables d'environnement (toutes optionnelles) :
    STT_MODEL_SIZE   tiny | base | small | medium | large-v3   (défaut : small)
    STT_LANGUAGE     "fr", "en"... ; vide = détection automatique
    STT_DEVICE       cuda | cpu ; vide = détection automatique
    STT_COMPUTE_TYPE float16 | int8 | ... ; vide = choix automatique
"""

# Fichier : app/core/audio_model_def.py
import asyncio
import io
import os
import threading
from dataclasses import dataclass
from typing import Optional

import ctranslate2
from faster_whisper import WhisperModel

STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "small")
STT_LANGUAGE = os.getenv("STT_LANGUAGE") or None
STT_DEVICE = os.getenv("STT_DEVICE") or None
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE") or None


class TranscriptionError(RuntimeError):
    """Levée quand la transcription échoue."""


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float


# --------------------------------------------------------------------------
# Modèle (chargé une seule fois, au premier usage)
# --------------------------------------------------------------------------
_model: Optional[WhisperModel] = None
_model_lock = threading.Lock()


def get_stt_model() -> WhisperModel:
    """Retourne le modèle partagé, en le chargeant au premier appel."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                has_cuda = ctranslate2.get_cuda_device_count() > 0
                device = STT_DEVICE or ("cuda" if has_cuda else "cpu")
                compute_type = STT_COMPUTE_TYPE or (
                    "float16" if device == "cuda" else "int8"
                )
                print(
                    f"Chargement faster-whisper ({STT_MODEL_SIZE}, "
                    f"{device}, {compute_type}) en mémoire..."
                )
                _model = WhisperModel(
                    STT_MODEL_SIZE, device=device, compute_type=compute_type
                )
                print("Modèle STT chargé avec succès !")
    return _model


# --------------------------------------------------------------------------
# Transcription purement en mémoire
# --------------------------------------------------------------------------
def transcribe_memory_sync(
    audio_buffer: io.BytesIO, language: Optional[str] = None
) -> TranscriptionResult:
    """Transcrit un flux audio binaire depuis la mémoire (bloquant)."""
    try:
        model = get_stt_model()
        # faster-whisper accepte directement un objet BytesIO ou un chemin
        segments, info = model.transcribe(
            audio_buffer,
            language=language or STT_LANGUAGE,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return TranscriptionResult(
            text=text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
        )
    except Exception as e:
        raise TranscriptionError(f"Échec de la transcription en mémoire : {e}") from e


async def transcribe_audio_bytes(
    audio_bytes: bytes,
    language: Optional[str] = None,
) -> TranscriptionResult:
    """
    Version async : convertit les bytes reçus en flux mémoire (BytesIO)
    et délègue le traitement à un thread séparé.
    """
    audio_buffer = io.BytesIO(audio_bytes)
    return await asyncio.to_thread(transcribe_memory_sync, audio_buffer, language)
