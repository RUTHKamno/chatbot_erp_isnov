def determine_recommended_chart(cube_query: dict, data: list) -> dict:
    """
    Détermine le graphique idéal selon le framework d'Andrew Abela.
    """
    num_rows = len(data)
    dimensions = cube_query.get("dimensions", [])
    measures = cube_query.get("measures", [])
    time_dimensions = cube_query.get("timeDimensions", [])

    if num_rows == 0:
        return {"type": "empty", "reason": "Aucune donnée"}

    if num_rows == 1 and len(dimensions) == 0:
        return {
            "type": "kpi_card",
            "title": "Chiffre Clé",
            "reason": "Une seule valeur globale",
        }

    if len(time_dimensions) > 0:
        return {
            "type": "line",
            "reason": "Évolution chronologique / Tendance dans le temps",
        }

    if len(dimensions) >= 1:
        first_dim_key = list(data[0].keys())[0] if data else ""
        sample_labels = [str(row.get(first_dim_key, "")) for row in data[:5]]
        avg_label_length = sum(len(lbl) for lbl in sample_labels) / max(
            len(sample_labels), 1
        )

        if 2 <= num_rows <= 5 and avg_label_length < 15:
            return {
                "type": "donut",
                "reason": "Composition / Part d'un tout (<= 5 catégories)",
            }

        if avg_label_length > 15 or num_rows > 7:
            return {
                "type": "horizontal_bar",
                "reason": "Étiquettes longues ou nombre élevé de catégories (>7)",
            }

        return {
            "type": "vertical_bar",
            "reason": "Comparaison de catégories distinctes (<7)",
        }

    return {"type": "table", "reason": "Affichage tabulaire par défaut"}
