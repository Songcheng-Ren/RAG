#!/usr/bin/env python
import json
import os
import argparse
def convert(input_path, output_path, field):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                out.append({field: item})
            elif isinstance(item, dict):
                val = None
                for k in ('text', 'answer', 'target', 'expected_answer'):
                    if k in item and isinstance(item[k], str) and item[k].strip():
                        val = item[k]
                        break
                if val is not None:
                    out.append({field: val})
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='./datasets/aalcr_chunks.json')
    p.add_argument('--output', type=str, default='./datasets/aalcr_chunks_objects.json')
    p.add_argument('--field', type=str, default='text')
    a = p.parse_args()
    convert(a.input, a.output, a.field)
