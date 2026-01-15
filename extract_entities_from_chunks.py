import json
import os
import re
import sys

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

def collect_chunks(input_path: str):
    if input_path.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = f.read().replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"^\s*#\s*Chunk\s+\d+\s*$", data, flags=re.MULTILINE)
        chunks = [p.strip() for p in parts if p.strip()]
        if not chunks:
            chunks = [data.strip()] if data.strip() else []
        return chunks
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str) and item.strip():
                chunks.append(item.strip())
            elif isinstance(item, dict):
                for field in ["text", "answer", "expected_answer", "target", "context"]:
                    if field in item and isinstance(item[field], str) and item[field].strip():
                        chunks.append(item[field].strip())
                        break
    elif isinstance(data, dict):
        for field in ["text", "answer", "expected_answer", "target", "context"]:
            if field in data and isinstance(data[field], str) and data[field].strip():
                chunks.append(data[field].strip())
                break
    return chunks

def extract_entities_with_spacy(text: str, nlp):
    entities = set()
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ALLOWED_LABELS:
            t = ent.text.strip()
            if len(t) > 1:
                entities.add(t)
    return entities

def main():
    import argparse
    p = argparse.ArgumentParser(description="Extract entity list from chunks file")
    p.add_argument("--input", type=str, default="/export/home/rensongcheng.2001/RAG/datasets/processed/medqa_chunks_selected.txt")
    p.add_argument("--output", type=str, default="/export/home/rensongcheng.2001/RAG/datasets/processed/medqa_entities_selected.txt")
    p.add_argument("--max-chunks", type=int, default=None)
    a = p.parse_args()
    try:
        import spacy
    except Exception as e:
        print("spacy未安装:", e)
        sys.exit(1)
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception as e:
        print("加载en_core_web_sm失败:", e)
        print("安装: python -m spacy download en_core_web_sm")
        sys.exit(1)
    if not os.path.exists(a.input):
        print(f"输入文件不存在: {a.input}")
        print("示例可用路径:")
        print("/export/home/rensongcheng.2001/RAG/datasets/processed/aalcr_chunks_selected.json")
        print("/export/home/rensongcheng.2001/RAG/datasets/processed/medqa_chunks_selected.txt")
        print("/export/home/rensongcheng.2001/RAG/datasets/aalcr_chunks.json")
        print("/export/home/rensongcheng.2001/RAG/datasets/medqa_chunks.txt")
        sys.exit(1)
    chunks = collect_chunks(a.input)
    if isinstance(a.max_chunks, int) and a.max_chunks > 0:
        chunks = chunks[:a.max_chunks]
    entities = set()
    for text in chunks:
        if isinstance(text, str) and text:
            sample = text[:2000] if len(text) > 2000 else text
            ents = extract_entities_with_spacy(sample, nlp)
            entities.update(ents)
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as f:
        for e in sorted(entities):
            f.write(e + "\n")
    print(f"{a.input} -> {a.output} ({len(entities)})")

if __name__ == "__main__":
    main()
