import json
import os
import re

def load_entities(path):
    entities = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                entities.append(t)
    return entities

def collect_chunks_from_file(input_path):
    if input_path.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            data = f.read().replace("\r\n", "\n").replace("\r", "\n")
        parts = re.split(r"^\s*#\s*Chunk\s+\d+\s*$", data, flags=re.MULTILINE)
        chunks = [p.strip() for p in parts if p.strip()]
        if not chunks:
            # fallback: treat whole file as one chunk
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

def filter_chunks_by_entities(entities, chunks):
    stop = {
        "the","a","an","of","and","or","in","on","for","to","with","without","from","by","as",
        "is","was","are","be","were","been","it","its","their","his","her","this","that","these","those"
    }
    ents = []
    for e in entities:
        t = e.lower().strip()
        if not t or t in stop:
            continue
        if t.isdigit():
            continue
        # skip very short single tokens and ambiguous abbreviations
        if " " not in t:
            if len(t) <= 2:
                continue
            if re.fullmatch(r"[a-z]{2}", t):
                continue
            if re.fullmatch(r"[a-z]{1,2}\d+", t):
                continue
        ents.append(t)
    ents = sorted(set(ents), key=len, reverse=True)
    robust_ents = []
    short_ents = []
    for t in ents:
        if (" " in t) or (not t.isalpha()) or (len(t) >= 3):
            robust_ents.append(t)
        else:
            short_ents.append(t)
    robust_patterns = []
    for t in robust_ents:
        if (" " in t) or (not t.isalpha()):
            robust_patterns.append(re.compile(re.escape(t)))
        else:
            robust_patterns.append(re.compile(r"\b" + re.escape(t) + r"\b"))
    short_patterns = [re.compile(r"\b" + re.escape(t) + r"\b") for t in short_ents]
    result = []
    seen = set()
    min_hits = 1
    for c in chunks:
        cl = c.lower()
        matched_total = 0
        matched_robust = 0
        for p in robust_patterns:
            if p.search(cl):
                matched_total += 1
                matched_robust += 1
        for p in short_patterns:
            if p.search(cl):
                matched_total += 1
        if matched_robust >= 2 or (matched_robust >= 1 and matched_total >= 3):
            if c not in seen:
                result.append(c)
                seen.add(c)
    return result

def save_mapping(mapping, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for e, c in mapping:
            f.write(f"{e}\t{c}\n")

def write_chunks_preserving_format(input_path, output_path, chunks):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if input_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks, 1):
                f.write(f"# Chunk {i}\n")
                f.write(c.strip() + "\n")

def process(entities_path, chunks_path, output_path):
    entities = load_entities(entities_path)
    chunks = collect_chunks_from_file(chunks_path)
    filtered = filter_chunks_by_entities(entities, chunks)
    write_chunks_preserving_format(chunks_path, output_path, filtered)
    print(f"{entities_path} + {chunks_path} -> {output_path} ({len(filtered)})")

def main():
    entities_base = "/export/home/rensongcheng.2001/RAG/datasets/processed"
    chunks_base = "/export/home/rensongcheng.2001/RAG/datasets"
    process(
        os.path.join(entities_base, "aa_lcr_prompt_entities.txt"),
        os.path.join(chunks_base, "aalcr_chunks.json"),
        os.path.join(entities_base, "aalcr_chunks_selected.json"),
    )
    process(
        os.path.join(entities_base, "medqa_prompt_entities.txt"),
        os.path.join(chunks_base, "medqa_chunks.txt"),
        os.path.join(entities_base, "medqa_chunks_selected.txt"),
    )

if __name__ == "__main__":
    main()
