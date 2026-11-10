import json
import os
from openai import OpenAI
from config import LLM_MODEL_NAME, OLLAMA_BASE_URL

def generate_dashboard_json(user_query: str, retrieved_contexts: list[str]) -> dict:
    combined_context = "\n".join([f"- {ctx}" for ctx in retrieved_contexts])
    
    system_prompt = """
    Tu es un expert en conception de tableaux de bord (dashboards).
    À partir de la question de l'utilisateur et des règles contextuelles de la thématique, 
    tu dois générer un JSON structurant les composants visuels attendus.
    
    Le JSON doit respecter strictement ce format :
    {
      "user_query": "...",
      "domain_intent": "...",
      "dashboard_title": "Titre du dashboard",
      "components": [
        {
          "type": "metric_card, line_chart, bar_chart, pie_chart ou table",
          "title": "Titre du composant",
          "description": "Ce que ce graphique ou KPI doit illustrer métierement"
        }
      ]
    }
    """

    user_prompt = f"""
    Question utilisateur : "{user_query}"
    Contexte métier de référence :
    {combined_context}
    
    Génère le JSON correspondant.
    """

    # Connexion à Ollama via le client OpenAI compatible
    client = OpenAI(
        api_key="ollama", # Ollama ne nécessite pas de vraie clé API
        base_url=OLLAMA_BASE_URL
    )
    
    print(f"[INFO] Appel au LLM local ({LLM_MODEL_NAME}) via Ollama...")
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        raw_content = response.choices[0].message.content
        return json.loads(raw_content)
        
    except Exception as e:
        print(f"[ERREUR] Échec de la génération du JSON par le LLM : {e}")
        return {"error": str(e), "user_query": user_query}