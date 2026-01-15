#!/usr/bin/env python
import os
import sys
import csv
import argparse
from typing import Set, Tuple

def read_relations_csv(input_csv: str) -> Set[Tuple[str, str]]:
    relations: Set[Tuple[str, str]] = set()
    if not os.path.exists(input_csv):
        print(f"关系文件不存在: {input_csv}")
        return relations
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_read = False
        for row in reader:
            if not header_read:
                header_read = True
                continue
            if not row or len(row) < 2:
                continue
            child = row[0].strip()
            parent = row[1].strip()
            if child and parent:
                relations.add((child, parent))
    return relations

def validate_relations_basic(relations: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    final_relations = set()
    for child, parent in relations:
        if not child or not parent:
            continue
        if child == parent:
            continue
        if (parent, child) in final_relations:
            continue
        final_relations.add((child, parent))
    return final_relations

def post_process_relations(relations: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    new_relations = set(relations)
    removed_pairs = set()
    for child, parent in relations:
        reverse_pair = (parent, child)
        if reverse_pair in relations:
            pair_key = tuple(sorted([child, parent]))
            if pair_key not in removed_pairs:
                removed_pairs.add(pair_key)
                if reverse_pair in new_relations:
                    new_relations.remove(reverse_pair)
    return new_relations

def write_subject_object_csv(relations: Set[Tuple[str, str]], output_csv: str):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "object"])
        for child, parent in sorted(relations):
            writer.writerow([parent, child])

def main():
    p = argparse.ArgumentParser(description="Filter relations CSV and output subject,object")
    p.add_argument("--input", type=str, default="/export/home/rensongcheng.2001/RAG/aalcr_entities_relations.csv")
    p.add_argument("--output", type=str, default="/export/home/rensongcheng.2001/RAG/aalcr_entities_relations_filtered.csv")
    a = p.parse_args()
    relations = read_relations_csv(a.input)
    if not relations:
        print("未读取到任何关系")
        sys.exit(1)
    relations = validate_relations_basic(relations)
    relations = post_process_relations(relations)
    if not relations:
        print("过滤后没有剩余关系")
        sys.exit(1)
    write_subject_object_csv(relations, a.output)
    print(f"{a.input} -> {a.output} ({len(relations)})")

if __name__ == "__main__":
    main()
