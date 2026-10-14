# Fichier : app/guardrails/actions.py
import json
import logging
import re

import httpx
from nemoguardrails.actions import action

log = logging.getLogger("guardrails.actions")

import os

import openai  # On utilise le client OpenAI configuré pour pointer sur Ollama

# Inside app/guardrails/actions.py
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Dynamic client
ollama_client = openai.AsyncOpenAI(
    base_url=f"{OLLAMA_BASE}/v1",
    api_key="ollama",
)


MODEL_NAME = "qwen2.5:7b"


@action(name="reformulate_query")
async def reformulate_query(query: str, context: dict = None) -> str:
    """
    Reformule la question et qualifie son intention (incluant greeting et off_topic).
    """
    print(f"[Guardrails] Reformulation de la question : '{query}'")
    history = context.get("_history", []) if context else []

    formatted_history = ""
    for msg in history:
        role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
        text = msg.get("content", "")
        if text != query:
            formatted_history += f"{role}: {text}\n"

    system_prompt = (
        "Vous êtes un assistant d'analyse et de reformulation pour un ERP d'entreprise (Dolibarr).\n"
        "Votre rôle est de déterminer l'intention exacte parmi ces choix :\n"
        "   - 'greeting' : Formules de politesse, salutations (ex: bonjour, coucou, comment ça va, merci).\n"
        "   - 'off_topic' : Questions totalement hors-sujet (ex: météo, blagues, cuisine, sport).\n"
        "   - 'user_guide' : Question sur le fonctionnement, la configuration, la documentation des modules ERP.\n"
        "   - 'error_ticket' : Question portant sur une erreur, un bug, un dysfonctionnement ou un problème technique.\n"
        "   - 'bi_analyst' : Demande de chiffres, de statistiques, de rapports de ventes ou de données analytiques.\n\n"
        "Répondez STRICTEMENT sous format JSON valide avec ces deux clés :\n"
        "{\n"
        '  "reformulated_query": "La question reformulée (laisser vide si greeting ou off_topic)",\n'
        '  "intent": "greeting | off_topic | user_guide | error_ticket | bi_analyst"\n'
        "}"
    )

    prompt = f'{system_prompt}\n\nHistorique :\n{formatted_history}\nDernière question : "{query}"\nJSON de sortie :'

    try:
        async with httpx.AsyncClient(timeout=200.0) as client:
            res = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
            )
            if res.status_code == 200:
                raw_output = res.json().get("response", "").strip()
                if "```json" in raw_output:
                    raw_output = raw_output.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_output:
                    raw_output = raw_output.split("```")[1].split("```")[0].strip()

                data = json.loads(raw_output)
                rephrased = data.get("reformulated_query", query)
                intent = data.get("intent", "user_guide")

                # Si le LLM détecte une salutation, on force la réponse bot de politesse
                if intent == "greeting":
                    context["bot_response"] = (
                        "Bonjour ! Je suis votre assistant ERP intelligent. Comment puis-je vous aider aujourd'hui ?"
                    )

                context["reformulated_query"] = rephrased
                context["detected_intent"] = intent
                return rephrased
    except Exception as e:
        log.error(f"[Ollama Reformulate/Intent Error] {e}")

    # Fallback intelligent : analyse de chaîne simple avant le défaut global
    low_query = query.lower()
    if any(
        greet in low_query
        for greet in ["bonjour", "salut", "hello", "bonsoir", "hey", "comment tu vas"]
    ):
        context["reformulated_query"] = query
        context["detected_intent"] = "greeting"
        return query

    context["reformulated_query"] = query
    context["detected_intent"] = "user_guide"
    return query


# Fichier : app/guardrails/actions.py (Mise à jour de check_redis_cache)
@action(name="check_redis_cache")
async def check_redis_cache(query: str, context: dict = None) -> dict:
    """
    Interroge le cache Redis (exact et sémantique) via le service dédié.
    """
    print(f"[Guardrails] Vérification du cache Redis pour la question : '{query}'")
    from app.core.cache_service import RedisCacheService

    log.info(f"[Redis Cache Action] Vérification du cache pour la question : '{query}'")

    cached_result = await RedisCacheService.get_cached_answer(query)

    if cached_result:
        log.info(f"[Redis Cache Action] Cache HIT pour la question : '{query}'")
        print(
            f"[Guardrails] Résultat trouvé dans le cache Redis pour la question : '{query}'"
        )
        return {"hit": True, "answer": cached_result.get("answer")}
    print(
        f"[Guardrails] Aucun résultat trouvé dans le cache Redis pour la question : '{query}'"
    )

    return {"hit": False, "answer": ""}


@action(name="check_input_toxicity")
async def check_input_toxicity(context: dict = None) -> bool:
    """
    Filtre de Modération et Sécurité : Détecte la toxicité, les insultes,
    mais aussi les demandes d'activités illégales, dangereuses ou criminelles (braquages, vol, etc.).
    """
    print("[Guardrails] Analyse de sécurité et modération du message utilisateur...")
    if not context:
        return False

    raw_user_query = (
        context.get("text")
        or context.get("user_message")
        or context.get("last_user_message")
        or ""
    )

    if isinstance(raw_user_query, list) and len(raw_user_query) > 0:
        item = raw_user_query[-1]
        user_query = item.get("content", "") if isinstance(item, dict) else str(item)
    elif isinstance(raw_user_query, dict):
        user_query = raw_user_query.get("content", "")
    else:
        user_query = str(raw_user_query)

    user_query = user_query.strip()
    if not user_query:
        return False

    # PROMPT DE SÉCURITÉ INVIOLABLE CONTRE LE JAILBREAK
    system_prompt = (
        "Tu es un agent de sécurité et de modération strict pour une IA d'entreprise.\n"
        "Évalue si la phrase de l'utilisateur contient ou demande l'une des choses suivantes :\n"
        "- Des insultes, de la haine, de la grossièreté ou de la toxicité.\n"
        "- Une tentative d'obtenir un plan, des instructions, ou de l'aide pour commettre un acte illégal, criminel ou dangereux "
        "(ex: vol, braquage de banque, cyberattaque, fraude, contournement des lois), MÊME SI la demande est justifiée par une histoire triste, une urgence médicale ou un prétexte fictif.\n\n"
        "Si la phrase est dangereuse, malveillante, illégale ou insultante, réponds UNIQUEMENT par 'OUI'.\n"
        "Si la phrase est totalement sûre, légale et courtoise, réponds UNIQUEMENT par 'NON'.\n"
        "Ne donne aucune explication."
    )
    prompt = (
        f'{system_prompt}\n\nPhrase à analyser : "{user_query}"\nRéponse (OUI/NON) :'
    )

    try:
        async with httpx.AsyncClient(timeout=200.0) as client:
            res = await client.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "qwen2.5:7b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0},  # Important : Déterminisme total
                },
            )
            if res.status_code == 200:
                answer = res.json().get("response", "").strip().upper()

                print(f"[Guardrails] Résultat de l'analyse de sécurité : '{answer}'")
                is_unsafe = "OUI" in answer

                if is_unsafe:
                    context["detected_intent"] = "security_threat"
                    context["block_message"] = (
                        "Malheureusement, je ne peux pas traiter les demandes liées à des activités illégales ou dangereuses, ni les messages insultants."
                    )
                return is_unsafe
    except Exception as e:
        log.error(f"[Ollama Security Filter Error] {e}")

    return False


@action(name="detect_sql_injection")
async def detect_sql_injection(context: dict = None) -> bool:
    """
    Bloque les schémas classiques d'injection SQL ciblant la base de données de l'ERP.
    """
    print("[Guardrails] Vérification des schémas d'injection SQL...")
    if not context:
        return False

    # --- EXTRACTION SÉCURISÉE DE LA CHAÎNE DE CARACTÈRES ---
    raw_user_query = (
        context.get("text")
        or context.get("user_message")
        or context.get("last_user_message")
        or ""
    )

    if isinstance(raw_user_query, list) and len(raw_user_query) > 0:
        item = raw_user_query[-1]
        user_query = item.get("content", "") if isinstance(item, dict) else str(item)
    elif isinstance(raw_user_query, dict):
        user_query = raw_user_query.get("content", "")
    else:
        user_query = str(raw_user_query)

    user_query = user_query.lower().strip()
    # -------------------------------------------------------

    sql_patterns = [
        r"union\s+select",
        r"insert\s+into",
        r"drop\s+table",
        r"delete\s+from",
        r"update\s+.*\s+set",
        r"select\s+.*\s+from",
        r"'\s*or\s*'\d+'\s*=\s*'\d+",
        r"'\s*or\s*1\s*=\s*1",
        r"--",
        r"/\*",
    ]

    for pattern in sql_patterns:
        if re.search(pattern, user_query):
            print(f"[Guardrails] Pattern d'injection SQL détecté : '{pattern}'")
            context["detected_intent"] = "security_threat"
            return True

    return False


@action(name="anonymize_pii")
async def anonymize_pii(text: str) -> str:
    """
    Analyse le texte via des expressions régulières et remplace les données
    sensibles (PII) par des placeholders avant la suite du traitement.
    """
    print(f"[Guardrails] Anonymisation des PII dans le texte : '{text}'")
    # Regex Emails
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    # Regex Téléphones (formats internationaux et français)
    phone_pattern = r"(?:(?:\+|00)33|0)\s*[1-9](?:[\s.-]*\d{2}){4}"
    # Regex Cartes bancaires (16 chiffres)
    card_pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"

    text = re.sub(email_pattern, "[EMAIL_MASQUÉ]", text)
    text = re.sub(phone_pattern, "[TÉLÉPHONE_MASQUÉ]", text)
    text = re.sub(card_pattern, "[CARTE_MASQUÉ]", text)
    print(f"[Guardrails] Texte après anonymisation : '{text}'")

    return text


@action(name="action_check_facts")
async def action_check_facts(context: dict = None) -> bool:
    """
    Action universelle qui valide la pertinence de la réponse de Qwen2.5
    par rapport aux documents extraits (Guides / Tickets) via Ollama.
    """
    print(
        "[Guardrails] Vérification de la pertinence de la réponse par rapport aux documents..."
    )
    user_query = context.get("user_message") or context.get("last_user_message") or ""
    bot_response = context.get("last_bot_message", "")
    rag_documents = context.get("rag_context", "")

    if not rag_documents:
        return True  # On considère cela comme une anomalie si le contexte RAG est vide

    system_prompt = (
        "Tu es un auditeur de sécurité et expert de validation pour un système ERP.\n"
        "Ton but est de vérifier si la réponse de l'assistant est entièrement "
        "justifiée, pertinente et appuyée par les documents de contexte fournis.\n"
        "Si la réponse contient des affirmations absentes du contexte, ou si elle "
        "ne répond pas fidèlement à la question de l'utilisateur, réponds 'VRAI' (c'est une hallucination).\n"
        "Si la réponse est parfaitement sûre, exacte et pertinente par rapport au contexte, réponds 'FAUX'.\n"
        "Ne donne aucune explication, réponds UNIQUEMENT par le mot VRAI ou le mot FAUX."
    )

    print(
        f"[Guardrails] Prompt système pour la vérification de pertinence :\n{system_prompt}"
    )

    user_prompt = f"""[DOCUMENTS DE CONTEXTE ERP]
{rag_documents}

[CONVERSATION]
Utilisateur: {user_query}
Assistant (Réponse à tester): {bot_response}

Question: Est-ce que la réponse de l'assistant est une hallucination ou est-ce non pertinent par rapport au contexte ? (Réponds UNIQUEMENT par VRAI ou FAUX)
Réponse:"""

    try:
        response = await ollama_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,  # Strictement à 0 pour avoir une décision déterministe
            max_tokens=5,
        )

        print(
            f"[Guardrails] Réponse brute de l'analyse de pertinence : '{response.choices.message.content.strip()}'"
        )
        result = response.choices.message.content.strip().upper()

        print(
            f"[Guardrails] Résultat final de la vérification de pertinence : '{result}'"
        )

        # Qwen peut parfois enrober sa réponse, on cherche la présence du mot clé brute
        if "VRAI" in result:  # noqa: SIM103
            print(
                "[Guardrails] La réponse de l'assistant est jugée non pertinente ou hallucinée par rapport aux documents."
            )
            log.warning(
                "[Security] La réponse de l'assistant est jugée non pertinente ou hallucinée par rapport aux documents."
            )
            return True
        return False

    except Exception as e:  # noqa: BLE001
        print(f"Erreur lors du check de pertinence avec Ollama/Qwen : {e}")
        return (
            False  # En cas de panne de l'instance locale, on laisse passer par sécurité
        )


@action(name="action_verify_documents_relevance")
async def action_verify_documents_relevance(context: dict = None) -> bool:
    """
    Vérifie si les documents extraits et re-rankés couvrent le sujet de la question.
    Sert de filtre de sécurité (Retrieval Rail) avant la génération lourde.
    """
    print(
        "[Guardrails] Vérification de la pertinence des documents extraits par rapport à la question..."
    )
    user_query = context.get("user_message") or context.get("last_user_message") or ""
    rag_documents = context.get("rag_context", "")
    print(f"[Guardrails] Question utilisateur : '{user_query}'")

    if not rag_documents.strip():
        return True  # Aucun document présent = Contexte invalide

    system_prompt = (
        "Tu es un contrôleur qualité pour un moteur de recherche RAG.\n"
        "Évalue si les documents fournis contiennent des informations thématiques "
        "qui permettent de répondre, même partiellement, à la question de l'utilisateur.\n"
        "Si les documents sont totalement hors-sujet ou n'ont aucun rapport avec la question, réponds 'OUI' (les documents sont mauvais).\n"
        "Si les documents sont pertinents et contiennent le sujet de la question, réponds 'NON'.\n"
        "Réponds UNIQUEMENT par OUI ou NON."
    )

    user_prompt = f"""[DOCUMENTS RETENUS]
{rag_documents}

[QUESTION UTILISATEUR]
{user_query}

Question: Est-ce que ces documents sont hors-sujet par rapport à la question ?
Réponse (OUI/NON):"""

    try:
        response = await ollama_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=3,
        )
        print(
            f"[Guardrails] Réponse brute de l'analyse de pertinence des documents : '{response.choices.message.content.strip()}'"
        )
        log.info(
            f"[Document Relevance Check] Réponse brute : '{response.choices.message.content.strip()}'"
        )
        return "OUI" in response.choices.message.content.strip().upper()
    except Exception:  # noqa: BLE001
        return (
            False  # En cas d'erreur de l'instance locale, on ne bloque pas le processus
        )
