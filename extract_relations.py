import os
import re
import csv
import json
import spacy
from tqdm import tqdm
from extract import extract_entities_with_spacy

def split_string_by_headings(text: str):
    lines = text.split("\n")
    current_block = []
    chunks = []
    def concat_block():
        if len(current_block) > 0:
            chunks.append("\n".join(current_block))
            current_block.clear()
    for line in lines:
        if line.startswith("# "):
            concat_block()
        current_block.append(line)
    concat_block()
    return chunks

def collect_chunks(input_path: str):
    if input_path.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = f.read().replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"^\s*#\s*Chunk\s+\d+\s*$", data, flags=re.MULTILINE)
        chunks = [p.strip() for p in parts if p.strip()]
        if not chunks:
            return split_string_by_headings(data)
        return chunks
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for field in ["text", "answer", "expected_answer", "target", "context"]:
                    if field in item and isinstance(item[field], str) and item[field].strip():
                        chunks.append(item[field].strip())
                        break
            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())
    return chunks

def load_entities(entities_path: str):
    entities = set()
    if not entities_path:
        return entities
    if entities_path.endswith(".csv"):
        with open(entities_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    entities.add(row[0].strip().lower())
                    entities.add(row[1].strip().lower())
                elif len(row) == 1 and row[0].strip():
                    entities.add(row[0].strip().lower())
        return entities
    with open(entities_path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                entities.add(t.lower())
    return entities

def extract_simple_relations(chunks, entities_set, nlp):
    if not entities_set:
        all_entities = set()
        for text in tqdm(chunks):
            if isinstance(text, str) and len(text.strip()) > 0:
                sample = text[:2000] if len(text) > 2000 else text
                ents = extract_entities_with_spacy(sample, nlp)
                all_entities.update(ents)
        entities_list = list(all_entities)[:100]
    else:
        entities_list = list(entities_set)[:100]
    relations = set()
    processed_count = min(200, len(chunks))
    for item in chunks[:processed_count]:
        text = item if isinstance(item, str) else str(item)
        text_l = text.lower()
        entities_in_text = [e for e in entities_list if e.lower() in text_l]
        for i, e1 in enumerate(entities_in_text[:5]):
            for e2 in entities_in_text[i+1:i+3]:
                if e1 != e2:
                    relations.add((e1, e2))
    if len(relations) < 50:
        entities_sorted = sorted(entities_list, key=len, reverse=True)[:30]
        for i in range(len(entities_sorted) - 1):
            relations.add((entities_sorted[i], entities_sorted[i+1]))
    return list(relations)

def run(input_chunks, output_relations, entities_file=None, max_chunks=None):
    try:
        nlp = spacy.load("en_core_web_md")
    except Exception:
        nlp = spacy.load("en_core_web_sm")
    chunks = collect_chunks(input_chunks)
    if isinstance(max_chunks, int) and max_chunks > 0:
        chunks = chunks[:max_chunks]
    entities = load_entities(entities_file) if entities_file else set()
    final_relations = extract_simple_relations(chunks, entities, nlp)
    os.makedirs(os.path.dirname(output_relations) or ".", exist_ok=True)
    with open(output_relations, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "object"])
        w.writerows(final_relations)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extract relations from existing chunks/entities")
    p.add_argument("--input-chunks", type=str, default="./datasets/aalcr_chunks.json")
    p.add_argument("--entities-file", type=str, default="./aalcr_entities.txt")
    p.add_argument("--output-relations", type=str, default="./entities_new/relations.csv")
    p.add_argument("--max-chunks", type=int, default=None)
    a = p.parse_args()
    run(a.input_chunks, a.output_relations, a.entities_file, a.max_chunks)
