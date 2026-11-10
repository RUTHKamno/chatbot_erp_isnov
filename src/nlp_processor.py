import re

def clean_and_normalize_query(user_query: str) -> str:
    """
    Nettoie et normalise la question de l'utilisateur (minuscules, suppression 
    des caractères spéciaux superflus, normalisation des espaces).
    """
    if not user_query:
        return ""
    
    # Passage en minuscules
    text = user_query.lower()
    
    # Suppression de la ponctuation superflue tout en gardant les mots clés
    text = re.sub(r'[^\w\sàâäéèêëîïôöùûüç]', ' ', text)
    
    # Normalisation des espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def detect_domain_intent(normalized_query: str) -> str:
    """
    Détecte l'intention thématique (domaine) à partir de la question nettoyée.
    (Ex: 'stock', 'vente', 'finance', etc.)
    """
    # Mots-clés associés au domaine stock (extensible facilement)
    stock_keywords = ["stock", "stocks", "rupture", "inventaire", "entrepot", "produits", "marchandise", "quantite"]
    
    for kw in stock_keywords:
        if kw in normalized_query:
            return "stock"
            
    # Domaine par défaut ou générique si aucun mot-clé spécifique n'est trouvé
    return "general"

def process_user_query(user_query: str) -> dict:
    """
    Pipeline complet de traitement NLP de la question utilisateur.
    Renvoie la question nettoyée et le domaine détecté.
    """
    cleaned_query = clean_and_normalize_query(user_query)
    domain_intent = detect_domain_intent(cleaned_query)
    
    return {
        "original_query": user_query,
        "cleaned_query": cleaned_query,
        "domain_intent": domain_intent
    }

if __name__ == "__main__":
    # Test rapide du module NLP
    test_q = "Quel est l'état actuel de nos STOCKS et quels sont les produits en rupture imminente ?"
    result = process_user_query(test_q)
    print(f"Résultat du traitement NLP : {result}")