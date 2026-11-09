# Fichier : app/core/audio_processor.py
import io
import os
import tempfile
from pathlib import Path

from pydub import AudioSegment, effects

# Dossier temporaire dédié
TEMP_CLEAN_DIR = Path(tempfile.gettempdir()) / "stt_cleaned_audio"


def clean_and_save_audio(raw_audio_bytes: bytes, suffix: str = ".wav") -> str:
    """
    Prend l'audio brut du micro, le nettoie (volume + fréquences)
    et le sauvegarde dans un fichier temporaire optimisé pour Whisper.
    """
    TEMP_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Charger l'audio brut en mémoire avec Pydub
    # (Note: st.audio_input envoie souvent du WAV, du MP3 ou OGG Webm)
    raw_audio_stream = io.BytesIO(raw_audio_bytes)
    audio = AudioSegment.from_file(raw_audio_stream)

    # 2. TRAITEMENT 1 : Conversion aux normes idéales de Whisper
    # Whisper adore le Mono à 16000Hz (16kHz)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # 3. TRAITEMENT 2 : Normalisation du volume (Gain automatique)
    # Si l'utilisateur parle loin du micro, cela réhausse sa voix au maximum sans saturer
    audio = effects.normalize(audio)

    # 4. TRAITEMENT 3 : Filtre anti-bruit de fond (Passe-haut)
    # Supprime toutes les fréquences en dessous de 120Hz (bruits de ventilateur, vibrations)
    audio = audio.high_pass_filter(120)

    # 5. Sauvegarde dans le fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, dir=TEMP_CLEAN_DIR
    )  # noqa: SIM115
    temp_file_path = temp_file.name
    temp_file.close()  # Libère le verrou pour permettre l'écriture

    audio.export(temp_file_path, format="wav")

    return temp_file_path
