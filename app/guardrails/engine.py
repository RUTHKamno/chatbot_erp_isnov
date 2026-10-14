# app/guardrails/engine.py
import os

from nemoguardrails import LLMRails, RailsConfig

from app.guardrails.actions import (
    action_check_facts,  # Importez votre action universelle Ollama
)

# 1. Chargement de la configuration depuis le dossier
config_path = os.path.join(os.path.dirname(__file__))
print(f"[Guardrails] Chargement de la configuration depuis : {config_path}")
# pointe vers votre dossier contenant config.yml et flows.co
config = RailsConfig.from_path(config_path)

# 2. Instanciation unique du moteur de rails
rails = LLMRails(config)

# 3. Enregistrement de l'action universelle
rails.register_action(action_check_facts)
