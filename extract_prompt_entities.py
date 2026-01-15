import json
import os
import sys
from pathlib import Path

ALLOWED_LABELS = {
    "PERSON",
    "ORG",
    "GPE",
    "LOC",
    "EVENT",
    "WORK_OF_ART",
    "LAW",
    "LANGUAGE",
    "PRODUCT",
    "MONEY",
    "DATE",
    "TIME",
    "PERCENT",
    "PER",
    "MISC",
}

def extract_entities_with_spacy(text, nlp):
    entities = set()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ALLOWED_LABELS:
            t = ent.text.strip()
            if len(t) > 1:
                entities.add(t)
    return entities


def process_dataset(input_path, output_path, nlp, max_samples=100):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:max_samples]
    entities = set()
    for item in data:
        text = item.get("prompt", "")
        if not text:
            continue
        text_sample = text[:2000] if len(text) > 2000 else text
        ents = extract_entities_with_spacy(text_sample, nlp)
        entities.update(ents)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for e in sorted(entities):
            f.write(e + "\n")
    print(f"{input_path} -> {output_path} ({len(entities)})")


def main():
    try:
        import spacy
    except Exception as e:
        print("spacy未安装:", e)
        sys.exit(1)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print("加载en_core_web_sm失败:", e)
        print("安装模型命令: python -m spacy download en_core_web_sm")
        sys.exit(1)
    aa_path = "/export/home/rensongcheng.2001/RAG/datasets/processed/aa_lcr.json"
    medqa_path = "/export/home/rensongcheng.2001/RAG/datasets/processed/medqa.json"
    aa_out = "/export/home/rensongcheng.2001/RAG/datasets/processed/aa_lcr_prompt_entities.txt"
    medqa_out = "/export/home/rensongcheng.2001/RAG/datasets/processed/medqa_prompt_entities.txt"
    process_dataset(aa_path, aa_out, nlp, 100)
    process_dataset(medqa_path, medqa_out, nlp, 100)


if __name__ == "__main__":
    main()
