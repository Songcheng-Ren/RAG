#!/usr/bin/env python
"""
从数据集中提取实体和chunks（不构建实体树）
使用GitHub仓库的build_index.py方法提取chunks
使用spacy NER提取实体（与查询时方法一致）
"""

import sys
import json
import os
import csv
from pathlib import Path
from typing import List, Dict, Set, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_base.build_index import load_vec_db
import spacy


def split_string_by_headings(text: str):
    """
    从GitHub仓库的build_index.py复制
    按标题分割文本为chunks
    """
    lines = text.split("\n")
    current_block: list[str] = []
    chunks: list[str] = []

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


def collect_chunks_from_file(file_path: str):
    """
    从GitHub仓库的build_index.py复制
    从文件收集chunks
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
        return split_string_by_headings(data)


def collect_chunks_from_dir(dir: str):
    """
    从GitHub仓库的build_index.py复制
    从目录收集chunks
    """
    chunks: list[str] = []
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir, filename)
            chunks.extend(collect_chunks_from_file(file_path))
    return chunks


def collect_chunks(dir_or_file: str):
    """
    从GitHub仓库的build_index.py复制
    统一的chunks收集接口
    """
    if os.path.isdir(dir_or_file):
        return collect_chunks_from_dir(dir_or_file)
    return collect_chunks_from_file(dir_or_file)


def extract_entities_with_spacy(text: str, nlp) -> Set[str]:
    """
    使用spacy NER提取实体（与查询时的方法一致）
    """
    entities = set()
    doc = nlp(text)
    
    # 提取所有命名实体
    for ent in doc.ents:
        # 保留常见的实体类型
        if ent.label_ in [
            "PERSON", "ORG", "GPE", "LOC", "EVENT", "WORK_OF_ART",
            "LAW", "LANGUAGE", "PRODUCT", "MONEY", "DATE", "TIME", "PERCENT",
            "PER", "MISC",  # 中文模型标签
        ]:
            entity_text = ent.text.strip()
            if len(entity_text) > 1:
                entities.add(entity_text)
    
    return entities


def extract_chunks_from_medqa(dataset_path: str, max_samples: int = None) -> List[str]:
    """从MedQA数据集中提取chunks（答案文本）"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    chunks = []
    for item in dataset:
        # MedQA包含answer字段
        answer = item.get('answer', item.get('expected_answer', ''))
        if answer and len(answer.strip()) > 0:
            chunks.append(answer.strip())
    
    if max_samples:
        chunks = chunks[:max_samples]
    
    print(f"✓ 从MedQA数据集提取了 {len(chunks)} 个chunks")
    return chunks


def extract_chunks_from_dart(dataset_path: str, max_samples: int = None) -> Tuple[List[str], Dict[int, str]]:
    """从DART数据集中提取chunks（答案文本）和source信息
    
    Returns:
        (chunks, chunk_to_source): chunks列表和chunk_id到source的映射
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    chunks = []
    chunk_to_source = {}  # {chunk_id: source}
    
    for idx, item in enumerate(dataset):
        # DART包含answer字段
        answer = item.get('answer', item.get('expected_answer', ''))
        if answer and len(answer.strip()) > 0:
            chunks.append(answer.strip())
            # 记录source信息
            source = item.get('source', '')
            if source:
                chunk_to_source[len(chunks) - 1] = source
    
    if max_samples:
        chunks = chunks[:max_samples]
        # 也限制chunk_to_source
        chunk_to_source = {k: v for k, v in chunk_to_source.items() if k < max_samples}
    
    print(f"✓ 从DART数据集提取了 {len(chunks)} 个chunks")
    if chunk_to_source:
        unique_sources = set(chunk_to_source.values())
        print(f"  - 包含 {len(unique_sources)} 个不同的source: {list(unique_sources)[:5]}{'...' if len(unique_sources) > 5 else ''}")
    
    return chunks, chunk_to_source


def extract_chunks_from_triviaqa(dataset_path: str, max_samples: int = None) -> List[str]:
    """从TriviaQA数据集中提取chunks"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    chunks = []
    for item in dataset:
        # TriviaQA可能包含answer或context字段
        answer = item.get('answer', item.get('expected_answer', item.get('Answer', '')))
        context = item.get('context', item.get('Context', ''))
        text = answer if answer else context
        if text and len(text.strip()) > 0:
            chunks.append(text.strip())
    
    if max_samples:
        chunks = chunks[:max_samples]
    
    print(f"✓ 从TriviaQA数据集提取了 {len(chunks)} 个chunks")
    return chunks


def build_entities_file_from_dataset(
    dataset_path: str, 
    output_path: str, 
    dataset_type: str = "auto",
    max_samples: int = 500,
    chunks: List[str] = None
):
    """
    从数据集提取实体并建立关系（不构建实体树）
    使用spacy NER提取实体（与查询时的方法一致）
    
    Args:
        dataset_path: 数据集JSON文件路径或chunks文件路径（如果是MedQA从textbooks提取）
        output_path: 输出实体关系文件路径
        dataset_type: 数据集类型
        max_samples: 最大处理样本数
        chunks: 如果提供了chunks列表，直接从中提取实体（用于MedQA从textbooks提取的情况）
    """
    print(f"正在从数据集提取实体关系（数据集类型: {dataset_type}）...")
    
    # 加载spacy模型
    try:
        if dataset_type in ["medqa", "dart"]:
            nlp = spacy.load("en_core_web_sm")
            print("✓ 使用spacy英文模型 (en_core_web_sm)")
        else:
            nlp = spacy.load("en_core_web_sm")
            print("✓ 使用spacy英文模型 (en_core_web_sm)")
    except Exception as e:
        raise RuntimeError(f"加载spacy模型失败: {e}")
    
    # 如果提供了chunks，直接从中提取实体（MedQA从textbooks提取的情况）
    if chunks:
        print(f"从提供的chunks列表提取实体（共 {len(chunks)} 个chunks）...")
        all_entities = set()
        processed_count = 0
        for chunk in chunks:
            if chunk and len(chunk.strip()) > 0:
                # 限制chunk长度以避免处理过长文本
                text_sample = chunk[:2000] if len(chunk) > 2000 else chunk
                entities = extract_entities_with_spacy(text_sample, nlp)
                all_entities.update(entities)
                processed_count += 1
                
                if processed_count % 100 == 0:
                    print(f"  已处理 {processed_count} 个chunks，提取到 {len(all_entities)} 个唯一实体...")
        
        print(f"✓ 使用spacy NER提取了 {len(all_entities)} 个唯一实体")
        
        # 保存实体列表文件
        entities_output_path = output_path.replace('_entities.csv', '_entities_list.txt')
        os.makedirs(os.path.dirname(entities_output_path) or '.', exist_ok=True)
        with open(entities_output_path, 'w', encoding='utf-8') as f:
            for entity in sorted(all_entities):
                f.write(f"{entity}\n")
        print(f"✓ 实体列表已保存到: {entities_output_path} (共 {len(all_entities)} 个实体)")
        return entities_output_path
    
    # 否则从JSON文件提取
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if max_samples:
        dataset = dataset[:max_samples]
    
    all_entities = set()
    relations = set()
    
    # 根据数据集类型选择字段
    if dataset_type == "medqa":
        text_field = 'answer'
    elif dataset_type == "dart":
        text_field = 'answer'
    elif dataset_type == "triviaqa":
        text_field = 'answer'  # 或'context'
    else:
        # 自动检测
        text_field = None
        for item in dataset[:10]:
            if 'answer' in item:
                text_field = 'answer'
                break
            elif 'expected_answer' in item:
                text_field = 'expected_answer'
                break
            elif 'context' in item:
                text_field = 'context'
                break
    
    if not text_field:
        print("✗ 无法自动检测数据集字段，请指定dataset_type")
        return None
    
    # 使用spacy NER提取实体
    print(f"使用spacy NER提取实体（从字段: {text_field}）...")
    
    processed_count = 0
    for item in dataset:
        text = item.get(text_field, item.get('expected_answer', ''))
        if text:
            text_sample = text[:1000] if len(text) > 1000 else text
            entities = extract_entities_with_spacy(text_sample, nlp)
            all_entities.update(entities)
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"  已处理 {processed_count} 条，提取到 {len(all_entities)} 个唯一实体...")
    
    print(f"✓ 使用spacy NER提取了 {len(all_entities)} 个唯一实体")
    
    # 创建基于文本的简单关系
    print("创建基于共现关系的实体关系...")
    entities_list = list(all_entities)[:100]  # 限制实体数量
    
    # 从数据集中查找共现的实体
    processed_count = min(200, len(dataset))
    for item in dataset[:processed_count]:
        text = item.get(text_field, item.get('expected_answer', '')).lower()
        entities_in_text = [e for e in entities_list if e.lower() in text]
        
        # 创建共现实体之间的关系
        for i, e1 in enumerate(entities_in_text[:5]):
            for e2 in entities_in_text[i+1:i+3]:
                if e1 != e2:
                    relations.add((e1, e2))
    
    # 如果还是不够，创建一些层级关系
    if len(relations) < 50:
        print("实体关系数量不足，创建基于长度的层级关系...")
        entities_sorted = sorted(entities_list, key=len, reverse=True)[:30]
        for i in range(len(entities_sorted) - 1):
            relations.add((entities_sorted[i], entities_sorted[i+1]))
    
    # 保存实体列表文件
    entities_output_path = output_path.replace('_entities.csv', '_entities_list.txt')
    os.makedirs(os.path.dirname(entities_output_path) or '.', exist_ok=True)
    with open(entities_output_path, 'w', encoding='utf-8') as f:
        for entity in sorted(all_entities):
            f.write(f"{entity}\n")
    print(f"✓ 实体列表已保存到: {entities_output_path} (共 {len(all_entities)} 个实体)")
    
    # 不再保存实体关系文件
    print(f"✓ 提取完成，未保存实体关系（仅保存实体列表）")
    
    return entities_output_path


def prepare_chunks_file(chunks: List[str], output_file: str):
    """将chunks保存为文本文件，以便load_vec_db使用"""
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    # 将chunks合并为一个文档，使用标题分隔（按照GitHub的方法）
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"# Chunk {i}\n")
            f.write(chunk)
            f.write("\n\n")
    
    print(f"✓ Chunks已保存到: {output_file}")


def extract_and_save(dataset_path: str, dataset_type: str, output_dir: str = "./extracted_data", max_samples: int = None):
    """
    提取实体和chunks并保存（不构建实体树）
    使用spacy NER提取实体（与查询时方法一致）
    
    Args:
        dataset_path: 数据集JSON文件路径
        dataset_type: 数据集类型 ('medqa', 'dart', 'triviaqa', 'auto')
        output_dir: 输出目录
        max_samples: 最大处理样本数
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确定输出文件名和vec_db_key
    if dataset_type == "medqa" and os.path.isdir(dataset_path):
        dataset_name = "medqa"
    else:
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    entities_file = os.path.join(output_dir, f"{dataset_name}_entities.csv")
    chunks_file = os.path.join(output_dir, f"{dataset_name}_chunks.txt")
    vec_db_key = dataset_name
    
    print("=" * 80)
    print(f"提取 {dataset_type} 数据集的实体和chunks")
    print("=" * 80)
    
    # 1. 提取chunks（从正确的源提取）
    print(f"\n步骤1: 提取chunks...")
    chunk_to_source = {}  # 用于存储chunk_id到source的映射
    if dataset_type == "medqa":
        # MedQA: 从textbooks目录提取（每个txt文件对应一棵树）
        if not os.path.isdir(dataset_path):
            # 如果没有提供目录，尝试默认路径
            default_path = "/Users/zongyikun/Downloads/Med_data_clean/textbooks/en"
            if os.path.exists(default_path):
                dataset_path = default_path
                print(f"使用默认MedQA textbooks目录: {dataset_path}")
            else:
                raise ValueError(f"MedQA需要提供textbooks目录路径，当前路径不存在: {dataset_path}")
        
        print(f"从MedQA textbooks目录提取chunks: {dataset_path}")
        # 使用collect_chunks_from_dir提取chunks（用于实体提取）
        from rag_base.build_index import collect_chunks_from_dir
        chunks = collect_chunks_from_dir(dataset_path)
        num_files = len([f for f in os.listdir(dataset_path) if f.endswith('.txt')])
        print(f"✓ 从 {num_files} 个txt文件提取了 {len(chunks)} 个chunks（每个文件将构建一棵树）")
    elif dataset_type == "dart":
        chunks, chunk_to_source = extract_chunks_from_dart(dataset_path, max_samples)
    elif dataset_type == "triviaqa":
        chunks = extract_chunks_from_triviaqa(dataset_path, max_samples)
    else:
        # 自动检测
        print("自动检测数据集类型...")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            sample = json.load(f)
            if isinstance(sample, list) and len(sample) > 0:
                first_item = sample[0]
                if 'answer' in first_item:
                    chunks = extract_chunks_from_medqa(dataset_path, max_samples)
                elif 'context' in first_item:
                    chunks = extract_chunks_from_triviaqa(dataset_path, max_samples)
                else:
                    chunks = extract_chunks_from_medqa(dataset_path, max_samples)
            else:
                chunks = []
    
    if not chunks:
        print("✗ 未能提取到chunks")
        return
    
    # 2. 保存chunks为文本文件（用于实体提取等）
    print(f"\n步骤2: 保存chunks到文件...")
    prepare_chunks_file(chunks, chunks_file)
    
    # 3. 构建向量数据库（使用GitHub的方法）
    print(f"\n步骤3: 构建向量数据库 (key: {vec_db_key})...")
    print("注意：如果数据库已存在，将直接加载")
    
    try:
        # 根据数据集类型选择正确的数据源
        if dataset_type == "medqa" and os.path.isdir(dataset_path):
            # MedQA: 从textbooks目录直接构建（每个txt文件一棵树）
            print(f"  MedQA数据集：从textbooks目录构建向量数据库（每个txt文件将构建一棵树）")
            db = load_vec_db(vec_db_key, dataset_path)  # 使用目录路径
        elif dataset_type == "dart":
            # DART: 从JSON文件构建以保留source信息（每个source一棵树）
            print(f"  DART数据集：从JSON文件构建向量数据库（保留source信息，每个source将构建一棵树）")
            db = load_vec_db(vec_db_key, dataset_path)  # 使用JSON文件
        else:
            # 其他数据集使用chunks.txt
            db = load_vec_db(vec_db_key, chunks_file)
        print(f"✓ 向量数据库构建/加载完成")
    except Exception as e:
        print(f"✗ 构建向量数据库失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 提取实体关系（不构建实体树，使用spacy NER）
    print(f"\n步骤4: 提取实体关系（不构建实体树，使用spacy NER）...")
    # 对于MedQA，我们已经有了chunks，直接传递给build_entities_file_from_dataset
    if dataset_type == "medqa" and os.path.isdir(dataset_path):
        # MedQA: 使用已提取的chunks提取实体
        build_entities_file_from_dataset(chunks_file, entities_file, dataset_type, max_samples, chunks=chunks)
    else:
        # 其他数据集：从dataset_path（JSON文件）提取实体
        build_entities_file_from_dataset(dataset_path, entities_file, dataset_type, max_samples)
    
    print("\n" + "=" * 80)
    print("✓ 提取完成！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  - 实体关系文件: {entities_file}")
    print(f"  - Chunks文件: {chunks_file}")
    print(f"  - 向量数据库: vec_db_cache/{vec_db_key}.db")
    print(f"\n注意: 此脚本只提取实体和chunks，不构建实体树。")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从数据集提取实体和chunks（不构建实体树）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 提取MedQA数据集
  python extract_entities_and_chunks.py --dataset ./datasets/processed/medqa.json --dataset-type medqa
  
  # 提取DART数据集
  python extract_entities_and_chunks.py --dataset ./datasets/processed/dart.json --dataset-type dart
  
  # 提取TriviaQA数据集
  python extract_entities_and_chunks.py --dataset ./datasets/processed/triviaqa.json --dataset-type triviaqa
        """
    )
    
    parser.add_argument('--dataset', type=str, required=True,
                       help='数据集JSON文件路径')
    parser.add_argument('--dataset-type', type=str, 
                       choices=['medqa', 'dart', 'triviaqa', 'auto'],
                       default='auto',
                       help='数据集类型（默认：auto自动检测）')
    parser.add_argument('--output-dir', type=str, default='./extracted_data',
                       help='输出目录（默认：./extracted_data）')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='最大处理样本数（默认：None处理全部）')
    
    args = parser.parse_args()
    
    extract_and_save(
        dataset_path=args.dataset,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

