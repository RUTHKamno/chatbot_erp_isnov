import os
import tempfile

import ollama
import torch
from huggingface_hub import login
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Définis ton fournisseur ici ou via le terminal : "hf" ou "ollama"
# Par exemple dans ton terminal : export MODEL_PROVIDER="ollama"
PROVIDER = os.getenv("MODEL_PROVIDER", "ollama").lower()


def initialize_vlm_model():
    """
    Initialise et retourne les objets nécessaires selon le fournisseur choisi (HF ou Ollama).
    """
    if PROVIDER == "ollama":
        # Avec Ollama, pas de chargement lourd en VRAM dans le script Python,
        # c'est le service Ollama local qui gère le modèle.
        model_name = os.getenv(
            "OLLAMA_MODEL_VLM", "qwen2.5vl:7b"
        )  # Remplace par le nom exact de ton modèle installé dans ollama
        print(f"Mode Ollama activé. Utilisation du modèle local : {model_name}")
        return None, model_name

    else:
        # Mode Hugging Face classique
        if os.getenv("HF_TOKEN"):
            login(token=os.getenv("HF_TOKEN"))

        model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
        print(f"Chargement du modèle Hugging Face {model_id}...")

        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
        print("Modèle VLM Hugging Face chargé avec succès !")
        return processor, model


def analyze_image_with_qwen(
    pil_image: Image.Image, processor_or_none, model_or_name, prompt, PROVIDER
) -> str:
    """
    Analyse une image PIL en redirigeant l'appel soit vers Hugging Face, soit vers Ollama
    selon la configuration active.
    """
    """
    prompt_system = (
        "You are an expert technical documentation assistant. "
        "Analyze this software interface screenshot, document page, or slide image carefully. "
        "Describe step-by-step the procedure, user actions, buttons clicked (with their numbers or names), "
        "and the business process flow so it can be fully indexed and understood in a text-based RAG system."
    )
    """

    if PROVIDER == "ollama":
        model_name = model_or_name
        # Ollama a besoin d'un fichier image physique ou d'un flux temporaire pour l'analyse visuelle
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil_image.save(tmp.name)
            tmp_path = tmp.name

        try:
            response = ollama.chat(
                model=model_name,
                options={"temperature": 0.1},
                messages=[{"role": "user", "content": prompt, "images": [tmp_path]}],
            )
            return response["message"]["content"]
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    else:
        # Logique Hugging Face d'origine
        processor = processor_or_none
        model = model_or_name

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text], images=[pil_image], padding=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text
