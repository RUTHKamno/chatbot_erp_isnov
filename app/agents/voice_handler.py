# Fichier : app/agents/voice_handler.py
import os

import httpx

from app.core.audio_model_def import transcribe_audio_bytes
from app.graph.state import SMAState

from langgraph.types import Interrupt

# import { interrupt } from "@langchain/langgraph";


PROVIDER = os.getenv("MODEL_PROVIDER", "hf").lower()

"""
async def voice_processing_node(state: SMAState) -> dict:
    
    Gère l'entrée vocale : Transcription -> Reformulation LLM -> Demande de validation utilisateur.
    Si refusé 2 fois -> Force le passage en mode textuel manuel.
    
    input_mode = state.get("input_mode", "text")

    # Si l'utilisateur est en mode texte classique, on saute ce nœud
    if input_mode != "voice":
        return {}

    voice_retry_count = state.get("voice_retry_count", 0)
    validation_status = state.get("validation_status", "pending")

    # Si l'utilisateur a déjà validé, on passe la main à la suite du graphe
    if validation_status == "validated":
        return {}

    # Si l'utilisateur a rejeté 2 fois de suite
    if voice_retry_count >= 2:
        print(
            "[Voice Handler] 🛑 Limite de 2 essais vocaux non validés atteinte. Passage forcé en saisie manuelle."
        )
        return {
            "force_manual_input": True,
            "route": "blocked",
            "answer": "Je n'ai pas réussi à capturer correctement votre demande par la voix après deux essais. Veuillez s'il vous plaît saisir votre message manuellement.",
        }

    # Étape 1 : Récupération et Transcription du fichier audio stocké temporairement
    audio_path = state.get("audio_file_path")
    if not audio_path or not os.path.exists(audio_path):
        return {"error": "Fichier audio introuvable.", "route": "blocked"}

    raw_transcription = await transcribe_audio_file(audio_path)
    print(f"[Voice Handler] Transcription brute : '{raw_transcription}'")

    # Étape 2 : Reformulation intelligente via le LLM (Ollama ou HF configuré)
    reformulated = raw_transcription  # Fallback par défaut
    prompt = (
        "Tu es un assistant expert pour un ERP. Reformule cette transcription vocale brute "
        "en une question claire, polie et orientée métier, sans ajouter d'explications superflues :\n"
        f"'{raw_transcription}'\nQuestion reformulée :"
    )

    try:
        if PROVIDER == "ollama":
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                if res.status_code == 200:
                    reformulated = res.json().get("response", raw_transcription).strip()
        else:
            # Appel API HF ou modèle local HF équivalent
            pass
    except Exception as e:
        print(f"[Voice Reformulate Error] {e}")

    # Étape 3 : Arrêt (Stop) du graphe pour proposer la reformulation à l'utilisateur (Pattern Human-in-the-loop)
    print(
        f"[Voice Handler] Proposition de reformulation à l'utilisateur : '{reformulated}' (Essai {voice_retry_count + 1}/2)"
    )

    return {
        "voice_transcription": raw_transcription,
        "reformulated_query": reformulated,
        "validation_status": "pending",
        "route": "blocked",  # Stoppe le graphe pour attendre la validation front-end
        "answer": f"J'ai compris : « {reformulated} ». Est-ce correct ? (Validez ou réessayez par la voix)",
    }
"""


async def voice_processing_node(state: SMAState) -> dict:
    """
    Gère l'entrée vocale avec interruption native LangGraph (interrupt).
    """
    input_mode = state.get("input_mode", "text")
    if input_mode != "voice":
        return {}

    voice_retry_count = state.get("voice_retry_count", 0)

    # Si l'utilisateur a déjà échoué 2 fois
    if voice_retry_count >= 2:
        return {
            "force_manual_input": True,
            "route": "blocked",
            "answer": "Limite de 2 essais vocaux non validés atteinte. Veuillez saisir votre message manuellement.",
        }

    # 1. Transcription audio
    audio_path = state.get("audio_file_path")
    if not audio_path or not os.path.exists(audio_path):
        return {"error": "Fichier audio introuvable.", "route": "blocked"}

    raw_transcription = await transcribe_audio_bytes(audio_path)

    # 2. Reformulation via le LLM actif (Ollama / HF)
    reformulated = raw_transcription
    prompt = f"""Tu es un assistant expert pour un ERP. Reformule cette transcription vocale brute "
            "en une question claire, polie et orientée métier, sans ajouter d'explications superflues :\n"
            '{raw_transcription}'\n'"""

    try:
        if PROVIDER == "ollama":
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": os.getenv("OLLAMA_MODEL", "qwen2.5:7b"),
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                if res.status_code == 200:
                    reformulated = res.json().get("response", raw_transcription).strip()
    except Exception as e:
        print(f"[Voice Reformulate Error] {e}")

    # 3. INTERRUPT NATIVE : Le graphe se met en pause et envoie cette question au client
    # L'exécution s'arrête ici et attend que l'API reprenne le graphe avec la validation de l'utilisateur.
    user_response = Interrupt(
        {
            "type": "voice_validation_required",
            "reformulated_query": reformulated,
            "raw_transcription": raw_transcription,
            "retry_count": voice_retry_count,
        }
    )

    # --- LE GRAPHE REPREND ICI APRÈS LA RÉPONSE DE L'UTILISATEUR ---
    # user_response contient ce que l'API a renvoyé lors de la reprise (ex: {"action": "validate", "query": "..."})

    if isinstance(user_response, dict) and user_response.get("action") == "reject":
        # Incrémentation du compteur de retry et relance du nœud ou blocage
        return {
            "voice_retry_count": voice_retry_count + 1,
            "validation_status": "rejected",
            "route": "blocked",
            "answer": "Reformulation refusée. Veuillez réessayer.",
        }

    # Si validé, on injecte la question propre dans la suite du graphe
    final_query = (
        user_response.get("query", reformulated)
        if isinstance(user_response, dict)
        else reformulated
    )

    return {
        "voice_transcription": raw_transcription,
        "reformulated_query": final_query,
        "validation_status": "validated",
        "raw_query": final_query,  # Remplace la requête brute par la voix validée
    }
