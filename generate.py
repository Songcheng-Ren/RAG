#!/usr/bin/env python
"""从已有的entities和chunks生成父子关系文件

使用场景：
- 你已经有实体列表（entities）
- 你有chunks（文本数据）
- 需要使用LLM分析实体之间的父子关系（层次关系），生成关系文件

构建方式：
1. ✅ 加载已有的实体列表（必须提供）
2. ✅ 从每个chunk找出出现的实体集合（chunk_entities）
3. ✅ 对chunk_entities里的实体做两两组合 → 候选pair
4. ✅ 使用LLM判断候选对之间的父子关系（child, parent）

输出格式：
- CSV文件，包含表头：child,parent
- 每行一个父子关系：子实体,父实体

优势：
- 更高效：只对确实共现的实体对进行LLM判断
- 更准确：基于实际共现上下文进行关系判断
- 专注于层次关系：专门识别父子关系
"""

import sys
import json
import os
import csv
from pathlib import Path
from typing import Set, List, Tuple, Optional
import re
from collections import defaultdict
from dotenv import load_dotenv

# 确保在导入OpenAI之前加载.env
load_dotenv()

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_model_name():
    """从环境变量获取模型名称，如果没有则返回默认值"""
    return os.getenv("MODEL_NAME") or "gpt-3.5-turbo"


def load_entities_from_file(entities_file: str) -> Set[str]:
    """从文件加载实体列表
    
    支持格式：
    - TXT格式：每行一个实体（推荐）
    - CSV格式：第一列是实体
    - JSON格式：实体数组
    
    示例TXT格式：
        实体1
        实体2
        实体3
    """
    entities = set()
    
    if not os.path.exists(entities_file):
        print(f"✗ 实体文件不存在: {entities_file}")
        return entities
    
    with open(entities_file, 'r', encoding='utf-8') as f:
        # 尝试JSON格式
        try:
            data = json.load(f)
            if isinstance(data, list):
                entities.update([str(e).strip() for e in data if e])
                print(f"✓ 从JSON加载了 {len(entities)} 个实体")
                return entities
        except:
            pass
        
        # 尝试CSV或文本格式
        f.seek(0)
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # CSV格式：取第一列
            if ',' in line:
                entity = line.split(',')[0].strip()
            else:
                entity = line.strip()
            
            if entity:
                entities.add(entity)
    
    print(f"✓ 从文件加载了 {len(entities)} 个实体")
    return entities


def load_chunks_from_file(chunks_file: str) -> List[str]:
    """从文件加载chunks
    
    支持格式：
    - JSON格式（包含chunks的数组或对象）
    - 文本文件（每行一个chunk或按段落分隔）
    """
    chunks = []
    
    if not os.path.exists(chunks_file):
        print(f"✗ Chunks文件不存在: {chunks_file}")
        return chunks
    
    # 尝试JSON格式
    try:
        with open(chunks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        chunks.append(item)
                    elif isinstance(item, dict):
                        # 尝试常见字段
                        for field in ['text', 'content', 'chunk', 'answer', 'prompt']:
                            if field in item and isinstance(item[field], str):
                                chunks.append(item[field])
                                break
            elif isinstance(data, dict):
                # 尝试找到chunks数组
                for key in ['chunks', 'texts', 'data']:
                    if key in data and isinstance(data[key], list):
                        chunks.extend([str(c) for c in data[key] if c])
                        break
        
        if chunks:
            print(f"✓ 从JSON加载了 {len(chunks)} 个chunks")
            return chunks
    except Exception as e:
        print(f"⚠ JSON解析失败，尝试文本格式: {e}")
    
    # 尝试文本格式（支持# Chunk N格式）
    with open(chunks_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # 检查是否是# Chunk N格式
        if content.strip().startswith('# Chunk'):
            # 按# Chunk分割
            parts = re.split(r'# Chunk \d+\s*\n', content)
            chunks = [p.strip() for p in parts if p.strip()]
        else:
            # 按空行分割
            chunks = [c.strip() for c in content.split('\n\n') if c.strip()]
            if not chunks:
                # 按行分割
                chunks = [line.strip() for line in content.split('\n') if line.strip() and not line.strip().startswith('#')]
    
    print(f"✓ 从文本加载了 {len(chunks)} 个chunks")
    return chunks


def extract_entities_with_llm(
    client: OpenAI,
    chunks: List[str],
    model_name: str = "gpt-3.5-turbo",
    batch_size: int = 20,
    max_entities: int = 1000
) -> Set[str]:
    """使用LLM从chunks中提取实体
    
    注意：此函数当前未在主流程中使用，但保留作为可选功能。
    可用于从chunks中自动提取实体列表，而不是从文件加载。
    
    Args:
        client: OpenAI客户端
        chunks: Chunks列表
        model_name: LLM模型名称
        batch_size: 每批处理的chunks数量
        max_entities: 最多提取的实体数量
    """
    all_entities = set()
    
    print(f"\n使用LLM ({model_name}) 从chunks中提取实体...")
    print(f"Chunks总数: {len(chunks)}, 批次大小: {batch_size}")
    
    # 将chunks分批处理
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_chunks)} 个chunks)...")
        
        # 构建prompt
        chunks_text = ""
        for idx, chunk in enumerate(batch_chunks, 1):
            # 限制chunk长度
            chunk_preview = chunk[:300] if len(chunk) > 300 else chunk
            chunks_text += f"{idx}. {chunk_preview}\n"
        
        prompt = f"""You are an entity extraction expert. Please extract all meaningful entities from the following text chunks.

Text chunks list:
{chunks_text}

Task:
1. Extract all meaningful entities, including but not limited to:
   - Person names, place names, organization names
   - Technical terms, concepts
   - Important events, dates, numbers (if related to entities)
   - Other meaningful named entities
2. Entities should be complete, meaningful phrases or words
3. Remove duplicate entities
4. Do not extract overly generic words (such as "the", "is", "a", etc.)

Output format requirements:
- One entity per line
- Only output entity names, do not output any explanatory text
- Entity names should maintain original capitalization
- Do not output duplicate entities

Please directly output the entity list, one entity per line:"""

        try:
            # 检查是否使用ARK API
            is_ark_api = "ark.cn-beijing.volces.com" in str(client.base_url) if hasattr(client, 'base_url') else False
            
            if is_ark_api:
                # ARK API格式
                try:
                    response = client.responses.create(
                        model=model_name,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                    )
                    # ARK API返回格式处理
                    result_text = ""
                    if hasattr(response, 'output') and response.output:
                        if isinstance(response.output, list):
                            for item in response.output:
                                if hasattr(item, 'content') and item.content:
                                    if isinstance(item.content, list) and len(item.content) > 0:
                                        content_item = item.content[0]
                                        if hasattr(content_item, 'text'):
                                            result_text = content_item.text
                                            break
                                        elif isinstance(content_item, dict) and 'text' in content_item:
                                            result_text = content_item['text']
                                            break
                                elif hasattr(item, 'text') and item.text:
                                    result_text = item.text
                                    break
                    elif hasattr(response, 'choices') and response.choices:
                        result_text = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else str(response.choices[0])
                    elif hasattr(response, 'content'):
                        result_text = response.content
                    
                    if not result_text and hasattr(response, 'output_text'):
                        result_text = response.output_text
                    
                    if not result_text:
                        result_text = str(response)
                    
                    if not isinstance(result_text, str):
                        result_text = str(result_text)
                except AttributeError:
                    is_ark_api = False
            
            if not is_ark_api:
                # 标准OpenAI API格式
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an entity extraction expert, skilled at extracting meaningful entities from text. Please strictly follow the required output format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                result_text = response.choices[0].message.content.strip()
            
            # 解析结果
            batch_entities = set()
            for line in result_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 移除可能的编号（如"1. ", "2. "等）
                line = re.sub(r'^\d+[\.\)]\s*', '', line)
                
                # 移除可能的引号
                line = line.strip('"\'')
                
                if line:
                    batch_entities.add(line)
            
            all_entities.update(batch_entities)
            print(f"  从批次 {batch_num} 中提取了 {len(batch_entities)} 个实体（累计: {len(all_entities)}）")
            
            # 如果实体数已经足够，提前结束
            if len(all_entities) >= max_entities:
                print(f"已达到最大实体数限制 ({max_entities})，停止处理")
                break
                
        except Exception as e:
            print(f"  ⚠ 批次 {batch_num} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 限制实体数量
    if len(all_entities) > max_entities:
        all_entities = set(list(all_entities)[:max_entities])
    
    print(f"\n✓ LLM共提取了 {len(all_entities)} 个实体")
    return all_entities


def find_entities_in_text(text: str, entities: Set[str]) -> List[str]:
    """在文本中精确查找实体（使用单词边界匹配，避免误匹配）
    
    Args:
        text: 要搜索的文本
        entities: 实体集合
        
    Returns:
        在文本中找到的实体列表
    """
    found_entities = []
    text_lower = text.lower()
    
    for entity in entities:
        entity_lower = entity.lower().strip()
        if not entity_lower:
            continue
        
        # 使用正则表达式进行单词边界匹配，避免误匹配
        # 例如："cat" 不会匹配 "category"
        # 转义特殊字符
        entity_escaped = re.escape(entity_lower)
        # 使用单词边界 \b 来确保精确匹配
        pattern = r'\b' + entity_escaped + r'\b'
        
        if re.search(pattern, text_lower, re.IGNORECASE):
            found_entities.append(entity)
    
    return found_entities


def process_chunks_and_generate_relations_incremental(
    client: OpenAI,
    chunks: List[str],
    entities: Set[str],
    model_name: str = "gpt-3.5-turbo",
    batch_size: int = 50,
    max_relations: int = 500,
    output_file: str = None,
    save_interval: int = 100
) -> Set[Tuple[str, str]]:
    """边找候选对边判断关系，每处理一定数量就保存一次
    
    步骤：
    1. 从每个chunk找出出现的实体集合（chunk_entities）
    2. 对chunk_entities里的实体做两两组合 → 候选pair
    3. 立即使用LLM判断关系（使用当前chunk作为上下文）
    4. 每处理save_interval个chunk就保存一次结果
    
    Args:
        client: OpenAI客户端
        chunks: Chunks列表
        entities: 实体集合
        model_name: LLM模型名称
        batch_size: 每批处理的实体对数量
        max_relations: 最多生成的关系数
        output_file: 输出文件路径（用于增量保存）
        save_interval: 每处理多少个chunk保存一次
    
    Returns:
        父子关系集合，每个元组格式为 (子实体, 父实体)
    """
    relations = set()
    processed_pairs = set()  # 记录已处理的pair，避免重复处理
    pair_batch = []  # 当前批次待处理的pair和上下文
    
    print(f"\n边找候选对边判断关系（每处理 {save_interval} 个chunk保存一次）...")
    print(f"Chunks总数: {len(chunks)}, 批次大小: {batch_size}")
    
    # 确保输出目录存在
    if output_file:
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    
    def save_relations_to_file(relations_set: Set[Tuple[str, str]], file_path: str, append: bool = False):
        """保存关系到文件"""
        mode = 'w'  # 总是重新写入完整文件
        with open(file_path, mode, encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['child', 'parent'])
            # 写入所有关系
            for child, parent in sorted(relations_set):
                if child and parent:
                    writer.writerow([child, parent])
    
    def process_batch_with_llm(batch: List[Tuple[Tuple[str, str], str]]):
        """处理一批实体对，使用LLM判断关系"""
        if not batch:
            return set()
        
        # 构建候选对集合（用于验证LLM返回的关系）
        candidate_pairs_set = set()
        for (e1, e2), _ in batch:
            candidate_pairs_set.add(tuple(sorted([e1, e2])))
        
        # 构建prompt
        pairs_text = ""
        contexts_text = ""
        
        for idx, ((e1, e2), context) in enumerate(batch, 1):
            pairs_text += f"{idx}. ({e1}, {e2})\n"
            contexts_text += f"\n候选对 ({e1}, {e2}) 的上下文:\n  {context[:300]}...\n"
        
        prompt = f"""You are a knowledge graph expert specializing in hierarchical relationships. Please determine parent-child relationships between the following candidate entity pairs.

Candidate entity pairs list:
{pairs_text}
{contexts_text}

Task:
1. Based on the provided context information, determine whether a parent-child (hierarchical) relationship exists between each candidate entity pair
2. A parent-child relationship means:
   - One entity is a broader/general concept (parent)
   - The other entity is a more specific/sub-concept (child)
   - Examples: "动物" (parent) and "狗" (child), "编程语言" (parent) and "Python" (child)
3. Only output entity pairs that have clear hierarchical relationships
4. If two entities only co-occur by chance but have no hierarchical relationship, do not output them
5. For each pair, determine which entity is the parent and which is the child

Output format requirements:
- One relationship per line, format: child,parent
- child is the more specific entity, parent is the more general entity
- Only output pairs that have clear hierarchical relationships
- Do not output duplicate relationship pairs
- A child entity can have multiple parent entities (different hierarchical levels are allowed).
- Output all valid hierarchical relationships without any restrictions on the number of children per parent.
- If a candidate pair has no clear hierarchical relationship, do not output it

Please directly output parent-child relationships, one per line, format: child,parent
Do not output any explanatory text, only output the relationship list:"""

        try:
            # 检查是否使用ARK API
            is_ark_api = "ark.cn-beijing.volces.com" in str(client.base_url) if hasattr(client, 'base_url') else False
            
            if is_ark_api:
                # ARK API格式
                try:
                    response = client.responses.create(
                        model=model_name,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                    )
                    result_text = ""
                    if hasattr(response, 'output') and response.output:
                        if isinstance(response.output, list):
                            for item in response.output:
                                if hasattr(item, 'content') and item.content:
                                    if isinstance(item.content, list) and len(item.content) > 0:
                                        content_item = item.content[0]
                                        if hasattr(content_item, 'text'):
                                            result_text = content_item.text
                                            break
                                        elif isinstance(content_item, dict) and 'text' in content_item:
                                            result_text = content_item['text']
                                            break
                                elif hasattr(item, 'text') and item.text:
                                    result_text = item.text
                                    break
                    elif hasattr(response, 'choices') and response.choices:
                        result_text = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else str(response.choices[0])
                    elif hasattr(response, 'content'):
                        result_text = response.content
                    
                    if not result_text and hasattr(response, 'output_text'):
                        result_text = response.output_text
                    
                    if not result_text:
                        result_text = str(response)
                    
                    if not isinstance(result_text, str):
                        result_text = str(result_text)
                except AttributeError:
                    is_ark_api = False
            
            if not is_ark_api:
                # 标准OpenAI API格式
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a knowledge graph expert, skilled at analyzing relationships between entities. Please strictly follow the required output format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                result_text = response.choices[0].message.content.strip()
            
            # 解析结果
            batch_relations = set()
            for line in result_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析实体对（格式：child,parent）
                if ',' in line:
                    parts = [p.strip() for p in line.split(',', 1)]
                    if len(parts) == 2 and parts[0] and parts[1]:
                        child, parent = parts[0], parts[1]
                        
                        # 验证关系是否合规
                        # 1. 检查子实体和父实体是否相同
                        if child == parent:
                            continue
                        # 2. 检查是否为空
                        if not child or not parent:
                            continue
                        # 3. 检查这个pair是否在候选对中（候选对是排序后的元组）
                        if tuple(sorted([child, parent])) not in candidate_pairs_set:
                            continue
                        # 4. 检查是否存在互相的双重关系（在同一批次内和已存在的关系中）
                        if (parent, child) not in relations and (parent, child) not in batch_relations:
                            batch_relations.add((child, parent))
            
            return batch_relations
            
        except Exception as e:
            print(f"  ⚠ LLM处理批次失败: {e}")
            return set()
    
    # 遍历chunks，边找边判断
    should_stop = False  # 标志变量，用于控制所有循环的退出
    for chunk_idx, chunk in enumerate(chunks):
        if (chunk_idx + 1) % 1 == 0:
            print(f"\r  [正在扫描] Chunk: {chunk_idx + 1} | 当前候选对池: {len(pair_batch)}/{batch_size} | 已找到关系: {len(relations)}", end="")
        if should_stop:
            break
        
        # 找出这个chunk中出现的所有实体（使用精确匹配，避免误匹配）
        chunk_entities = find_entities_in_text(chunk, entities)
        
        # 对chunk_entities里的实体做两两组合，立即加入批次
        for i in range(len(chunk_entities)):
            if should_stop:
                break
            for j in range(i + 1, len(chunk_entities)):
                if should_stop:
                    break
                e1, e2 = chunk_entities[i], chunk_entities[j]
                if e1 != e2:
                    pair = tuple(sorted([e1, e2]))
                    # 如果这个pair还没处理过，加入批次
                    if pair not in processed_pairs:
                        pair_batch.append((pair, chunk[:500]))  # 使用当前chunk作为上下文
                        processed_pairs.add(pair)
                        
                        # 当批次达到batch_size时，处理这批
                        if len(pair_batch) >= batch_size:
                            batch_relations = process_batch_with_llm(pair_batch)
                            relations.update(batch_relations)
                            pair_batch = []  # 清空批次
                            
                            if len(relations) >= max_relations:
                                print(f"已达到最大关系数限制 ({max_relations})，停止处理")
                                should_stop = True
                                break
        
        # 每处理save_interval个chunk就保存一次
        if (chunk_idx + 1) % save_interval == 0:
            if output_file:
                # 保存当前的所有关系（重新写入完整文件）
                save_relations_to_file(relations, output_file, append=False)
            print(f"  已处理 {chunk_idx + 1}/{len(chunks)} 个chunks，生成了 {len(relations)} 个关系（已保存）")
        
        # 检查是否达到最大关系数（在外层循环中也检查一次，确保及时退出）
        if len(relations) >= max_relations:
            should_stop = True
            break
    
    # 处理剩余的批次
    if pair_batch:
        batch_relations = process_batch_with_llm(pair_batch)
        relations.update(batch_relations)
    
    # 最终保存
    if output_file:
        save_relations_to_file(relations, output_file, append=False)  # 重新写入完整文件
    
    print(f"✓ 共处理了 {min(chunk_idx + 1, len(chunks))} 个chunks，生成了 {len(relations)} 个关系")
    return relations


def call_llm_for_hierarchical_relations(
    client: OpenAI,
    candidate_pairs: Set[Tuple[str, str]],
    pair_contexts: dict,
    model_name: str = "gpt-3.5-turbo",
    batch_size: int = 50,
    max_relations: int = 500,
    use_llm: bool = True
) -> Set[Tuple[str, str]]:
    """使用LLM判断候选实体对之间的父子关系（层次关系）
    
    注意：此函数当前未在主流程中使用，但保留作为可选功能。
    主流程使用的是 process_chunks_and_generate_relations_incremental() 函数。
    此函数可用于两阶段处理：先收集所有候选对，再批量判断关系。
    
    生成过程中不做任何限制，所有有效的关系都会被保留。
    筛选将在生成完成后进行。
    
    Args:
        client: OpenAI客户端
        candidate_pairs: 候选实体对集合（从chunks共现中提取）
        pair_contexts: 每个实体对的相关上下文（用于后续筛选）
        model_name: LLM模型名称
        batch_size: 每批处理的实体对数量
        max_relations: 最多生成的关系数
        use_llm: 是否使用LLM判断，如果False则直接返回所有候选对作为共现边
    
    Returns:
        父子关系集合，每个元组格式为 (子实体, 父实体)
    """
    if not use_llm:
        # 直接使用共现边，不经过LLM判断（无法确定父子关系，返回空）
        print(f"\n直接使用共现边，但无法确定父子关系，返回空集合")
        print(f"提示：请使用 --use-llm 来生成父子关系")
        return set()
    
    # 收集所有关系，不做限制
    relations = set()
    candidate_pairs_list = list(candidate_pairs)
    
    print(f"\n使用LLM ({model_name}) 判断候选实体对关系...")
    print(f"候选对总数: {len(candidate_pairs_list)}, 批次大小: {batch_size}")
    
    # 将候选对分批处理
    for i in range(0, len(candidate_pairs_list), batch_size):
        batch_pairs = candidate_pairs_list[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(candidate_pairs_list) + batch_size - 1) // batch_size
        
        print(f"\n处理批次 {batch_num}/{total_batches} ({len(batch_pairs)} 个候选对)...")
        
        # 构建prompt：列出候选对和它们的上下文
        pairs_text = ""
        contexts_text = ""
        
        for idx, (e1, e2) in enumerate(batch_pairs, 1):
            pairs_text += f"{idx}. ({e1}, {e2})\n"
            
            # 添加这个pair的上下文
            if (e1, e2) in pair_contexts and pair_contexts[(e1, e2)]:
                contexts = pair_contexts[(e1, e2)][:2]  # 每个pair最多2个上下文
                contexts_text += f"\n候选对 ({e1}, {e2}) 的上下文:\n"
                for ctx_idx, ctx in enumerate(contexts, 1):
                    contexts_text += f"  {ctx_idx}. {ctx[:300]}...\n"
        
        prompt = f"""You are a knowledge graph expert specializing in hierarchical relationships. Please determine parent-child relationships between the following candidate entity pairs.

Candidate entity pairs list:
{pairs_text}
{contexts_text}

Task:
1. Based on the provided context information, determine whether a parent-child (hierarchical) relationship exists between each candidate entity pair
2. A parent-child relationship means:
   - One entity is a broader/general concept (parent)
   - The other entity is a more specific/sub-concept (child)
   - Examples: "动物" (parent) and "狗" (child), "编程语言" (parent) and "Python" (child)
3. Only output entity pairs that have clear hierarchical relationships
4. If two entities only co-occur by chance but have no hierarchical relationship, do not output them
5. For each pair, determine which entity is the parent and which is the child

Output format requirements:
- One relationship per line, format: child,parent
- child is the more specific entity, parent is the more general entity
- Only output pairs that have clear hierarchical relationships
- Do not output duplicate relationship pairs
- A child entity can have multiple parent entities (different hierarchical levels are allowed).
- Output all valid hierarchical relationships without any restrictions on the number of children per parent.
- If a candidate pair has no clear hierarchical relationship, do not output it

Please directly output parent-child relationships, one per line, format: child,parent
Do not output any explanatory text, only output the relationship list:"""

        try:
            # 检查是否使用ARK API
            is_ark_api = "ark.cn-beijing.volces.com" in str(client.base_url) if hasattr(client, 'base_url') else False
            
            if is_ark_api:
                # ARK API格式
                try:
                    response = client.responses.create(
                        model=model_name,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                    )
                    # ARK API返回格式处理
                    # ARK API response format: output is a list
                    # [0] = ResponseReasoningItem (reasoning process)
                    # [1] = ResponseOutputMessage (actual answer with content)
                    result_text = ""
                    if hasattr(response, 'output') and response.output:
                        if isinstance(response.output, list):
                            # Look for ResponseOutputMessage (actual answer)
                            for item in response.output:
                                # Check if it's a message with content
                                if hasattr(item, 'content') and item.content:
                                    if isinstance(item.content, list) and len(item.content) > 0:
                                        content_item = item.content[0]
                                        # Extract text from ResponseOutputText
                                        if hasattr(content_item, 'text'):
                                            result_text = content_item.text
                                            break
                                        elif isinstance(content_item, dict) and 'text' in content_item:
                                            result_text = content_item['text']
                                            break
                                # Fallback: check for text attribute directly
                                elif hasattr(item, 'text') and item.text:
                                    result_text = item.text
                                    break
                    elif hasattr(response, 'choices') and response.choices:
                        # Standard OpenAI format fallback
                        result_text = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else str(response.choices[0])
                    elif hasattr(response, 'content'):
                        result_text = response.content
                    
                    # If still no text, try output_text attribute
                    if not result_text and hasattr(response, 'output_text'):
                        result_text = response.output_text
                    
                    # Last resort: convert to string
                    if not result_text:
                        result_text = str(response)
                    
                    # Ensure result_text is a string
                    if not isinstance(result_text, str):
                        result_text = str(result_text)
                except AttributeError:
                    # 如果responses.create不存在，回退到标准API
                    is_ark_api = False
            
            if not is_ark_api:
                # 标准OpenAI API格式
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a knowledge graph expert, skilled at analyzing relationships between entities. Please strictly follow the required output format."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3,  # 降低温度以获得更一致的结果
                    max_tokens=2000
                )
                result_text = response.choices[0].message.content.strip()
            
            # 解析结果（格式：child,parent）
            print(f"  [开始验证] 开始验证批次 {batch_num} 的关系合规性...")
            batch_new_relations = 0
            batch_invalid_relations = 0
            invalid_reasons = defaultdict(int)
            has_invalid = False  # 标记是否检测到不合规关系
            
            for line in result_text.split('\n'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # 解析实体对（格式：child,parent）
                if ',' in line:
                    parts = [p.strip() for p in line.split(',', 1)]
                    if len(parts) == 2 and parts[0] and parts[1]:
                        child, parent = parts[0], parts[1]
                        
                        # 验证关系是否合规
                        is_valid = True
                        reason = None
                        
                        # 1. 检查子实体和父实体是否相同
                        if child == parent:
                            is_valid = False
                            reason = "子实体和父实体相同"
                        
                        # 2. 检查是否为空字符串（去除空白后）
                        elif not child or not parent:
                            is_valid = False
                            reason = "实体为空"
                        
                        # 3. 检查这个pair是否在候选对中（候选对是排序后的元组）
                        elif tuple(sorted([child, parent])) not in candidate_pairs:
                            is_valid = False
                            reason = "不在候选对中"
                        
                        # 4. 检查是否存在互相的双重关系（如已有 (B, A)，则不允许 (A, B)）
                        elif (parent, child) in relations:
                            is_valid = False
                            reason = "存在互相的双重关系"
                        
                        if is_valid:
                            # 添加关系（不做限制）
                            relations.add((child, parent))
                            batch_new_relations += 1
                        else:
                            # [开始处理不合规] 首次检测到不合规关系时输出标记
                            if not has_invalid:
                                print(f"  [开始处理不合规] 检测到不合规关系，开始处理...")
                                has_invalid = True
                            
                            # [处理不合规] 标记不合规关系
                            batch_invalid_relations += 1
                            if reason:
                                invalid_reasons[reason] += 1
            
            print(f"  [验证完成] 批次 {batch_num} 验证完成：提取了 {batch_new_relations} 个新关系", end="")
            if batch_invalid_relations > 0:
                reason_str = ", ".join([f"{k}: {v}" for k, v in invalid_reasons.items()])
                print(f" | [处理不合规] 共处理了 {batch_invalid_relations} 个不合规关系: {reason_str}", end="")
            print()
            
            # 如果关系数已经足够，提前结束
            if len(relations) >= max_relations:
                print(f"已达到最大关系数限制 ({max_relations})，停止处理")
                break
                
        except Exception as e:
            print(f"  ⚠ 批次 {batch_num} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 限制关系数量
    if len(relations) > max_relations:
        relations = set(list(relations)[:max_relations])
    
    # 最终验证：再次检查是否有不合规的关系
    print(f"\n  [开始处理不合规] 开始最终验证，检查是否有不合规的关系...")
    final_relations = set()
    final_invalid = 0
    
    for child, parent in relations:
        # 最终验证
        if not child or not parent:
            final_invalid += 1
        elif child == parent:
            final_invalid += 1
        elif (parent, child) in final_relations:
            # 检查是否存在互相的双重关系
            final_invalid += 1
        else:
            final_relations.add((child, parent))
    
    if final_invalid > 0:
        print(f"  [处理不合规] 最终验证共处理了 {final_invalid} 个不合规关系，已移除")
    else:
        print(f"  [处理不合规] 最终验证未发现不合规关系")
    
    print(f"\n✓ LLM共生成 {len(final_relations)} 个父子关系（已过滤不合规关系，未筛选）")
    return final_relations


def post_process_relations(relations: Set[Tuple[str, str]]) -> Set[Tuple[str, str]]:
    """后处理关系：移除双重关系（矛盾的关系）
    
    双重关系是指：如果存在 (A, B) 和 (B, A)，这是矛盾的（A不能既是B的子实体，又是B的父实体）。
    这个处理不需要LLM，只是移除矛盾的关系。
    
    Args:
        relations: 原始关系集合，格式为 (child, parent)
    
    Returns:
        后处理后的关系集合（已移除双重关系）
    """
    print(f"\n[后处理] 对关系进行后处理：移除双重关系...")
    
    # 检查并移除双重关系
    # 双重关系：如果存在 (A, B) 和 (B, A)，这是矛盾的
    new_relations = set(relations)
    removed_pairs = set()  # 记录已处理的双重关系对，避免重复处理
    
    for child, parent in relations:
        reverse_pair = (parent, child)
        # 如果存在反向关系，这是双重关系，需要移除
        if reverse_pair in relations:
            # 使用排序后的pair作为唯一标识，避免重复处理
            pair_key = tuple(sorted([child, parent]))
            if pair_key not in removed_pairs:
                removed_pairs.add(pair_key)
                # 移除反向关系（保留第一个遇到的，移除第二个）
                if reverse_pair in new_relations:
                    new_relations.remove(reverse_pair)
    
    removed_count = len(removed_pairs)
    
    print(f"✓ 后处理完成：从 {len(relations)} 个关系处理到 {len(new_relations)} 个关系")
    if removed_count > 0:
        print(f"  移除了 {removed_count} 个双重关系（矛盾的关系）")
    
    return new_relations


def filter_relations_by_occurrence_count(
    relations: Set[Tuple[str, str]],
    top_k: int = 10
) -> Set[Tuple[str, str]]:
    """筛选关系：每个父实体只保留 top K 个子实体（按字母顺序）
    
    Args:
        relations: 所有关系集合，格式为 (child, parent)
        top_k: 每个父实体保留的子实体数量（默认：10）
    
    Returns:
        筛选后的关系集合
    """
    print(f"\n[筛选步骤] 对关系进行筛选：每个父实体保留 top {top_k} 个子实体")
    
    # 按父实体分组
    parent_to_children: dict[str, list[str]] = defaultdict(list)
    
    for child, parent in relations:
        parent_to_children[parent].append(child)
    
    # 对每个父实体的子实体按名称排序，取 top K
    filtered_relations = set()
    total_filtered = 0
    
    for parent, children in parent_to_children.items():
        # 按子实体名称排序
        children_sorted = sorted(children)
        
        # 取 top K
        top_children = children_sorted[:top_k]
        
        for child in top_children:
            filtered_relations.add((child, parent))
        
        if len(children) > top_k:
            total_filtered += len(children) - top_k
    
    print(f"✓ 筛选完成：从 {len(relations)} 个关系筛选到 {len(filtered_relations)} 个关系")
    if total_filtered > 0:
        print(f"  移除了 {total_filtered} 个关系（每个父实体只保留 top {top_k} 个子实体）")
    
    return filtered_relations


def filter_relations_by_top_children(
    relations: Set[Tuple[str, str]],
    pair_contexts: dict,
    top_k: int = 10
) -> Set[Tuple[str, str]]:
    """筛选关系：每个父实体只保留 top K 个子实体（根据关系出现频率）
    
    Args:
        relations: 所有关系集合，格式为 (child, parent)
        pair_contexts: 每个实体对的相关上下文（用于计算频率）
        top_k: 每个父实体保留的子实体数量（默认：10）
    
    Returns:
        筛选后的关系集合
    """
    print(f"\n[筛选步骤] 对关系进行筛选：每个父实体保留 top {top_k} 个子实体")
    
    # 按父实体分组，并计算每个子实体的评分（使用上下文数量作为频率）
    parent_to_children_scores: dict[str, list[Tuple[str, int]]] = defaultdict(list)
    
    for child, parent in relations:
        pair = tuple(sorted([child, parent]))
        # 使用上下文数量作为评分（关系出现频率）
        score = len(pair_contexts.get(pair, []))
        parent_to_children_scores[parent].append((child, score))
    
    # 对每个父实体的子实体按评分排序，取 top K
    filtered_relations = set()
    total_filtered = 0
    
    for parent, children_scores in parent_to_children_scores.items():
        # 按评分降序排序（评分相同则按子实体名称排序）
        children_scores.sort(key=lambda x: (-x[1], x[0]))
        
        # 取 top K
        top_children = children_scores[:top_k]
        
        for child, score in top_children:
            filtered_relations.add((child, parent))
        
        if len(children_scores) > top_k:
            total_filtered += len(children_scores) - top_k
    
    print(f"✓ 筛选完成：从 {len(relations)} 个关系筛选到 {len(filtered_relations)} 个关系")
    if total_filtered > 0:
        print(f"  移除了 {total_filtered} 个关系（每个父实体只保留 top {top_k} 个子实体）")
    
    return filtered_relations


def generate_hierarchical_relations(
    chunks_file: str,
    entities_file: str,
    output_relations_file: str,
    model_name: str = None,
    batch_size: int = 50,
    max_relations: int = 500,
    use_llm: bool = True,
    start_index: int = 0
) -> Optional[str]:
    """从已有的entities和chunks生成父子关系文件
    
    构建方式：
    1. 加载已有的实体列表（必须提供）
    2. 加载chunks
    3. 从每个chunk找出出现的实体集合（chunk_entities）
    4. 对chunk_entities里的实体做两两组合 → 候选pair
    5. 使用LLM判断候选对之间的父子关系（child, parent）
    6. 生成实体关系文件（CSV格式：child,parent）
    
    Args:
        chunks_file: Chunks文件路径
        entities_file: 实体列表文件路径（必须提供）
        output_relations_file: 输出的实体关系文件路径（CSV格式，每行：child,parent）
        model_name: LLM模型名称
        batch_size: 每批处理的实体对数量（默认：50）
        max_relations: 最多生成的关系数
        use_llm: 是否使用LLM判断，如果False则无法生成关系（必须为True）
    
    Returns:
        实体关系文件路径，如果失败则返回 None
    """
    print("=" * 80)
    print("从已有的entities和chunks生成父子关系文件")
    print("=" * 80)
    
    # 1. 加载实体列表（必须提供）
    if not entities_file or not os.path.exists(entities_file):
        print(f"✗ 错误: 实体文件不存在: {entities_file}")
        print("  请提供有效的实体列表文件")
        return None
    
    print(f"\n[步骤1] 加载实体列表: {entities_file}")
    entities = load_entities_from_file(entities_file)
    if not entities:
        print("✗ 无法加载实体列表")
        return None
    print(f"✓ 加载了 {len(entities)} 个实体")
    
    # 2. 加载chunks
    print(f"\n[步骤2] 加载chunks: {chunks_file}")
    chunks = load_chunks_from_file(chunks_file)
    if not chunks:
        print("✗ 无法加载chunks")
        return None
    if start_index and start_index > 0:
        start = min(max(0, int(start_index)), len(chunks))
        chunks = chunks[start:]
        print(f"✓ 加载了 {len(chunks)} 个chunks（从索引 {start} 开始）")
    else:
        print(f"✓ 加载了 {len(chunks)} 个chunks")
    
    # 3. 初始化LLM客户端
    if not use_llm:
        print("✗ 错误: 生成父子关系必须使用LLM，请设置 --use-llm")
        return None
    
    if not model_name:
        model_name = get_model_name()
        print(f"从环境变量读取模型名称: {model_name}")
    
    print(f"\n[步骤3] 初始化LLM客户端 (模型: {model_name})")
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"
    
    if not api_key:
        print("✗ 错误: 未找到API密钥，请设置 ARK_API_KEY 或 OPENAI_API_KEY 环境变量")
        return None
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    # 4. 边找候选对边判断关系（每处理100个chunk保存一次）
    print(f"\n[步骤4] 边找候选对边判断关系（增量处理）")
    relations = process_chunks_and_generate_relations_incremental(
        client,
        chunks,
        entities,
        model_name=model_name,
        batch_size=batch_size,
        max_relations=max_relations,
        output_file=output_relations_file,
        save_interval=100
    )
    
    if not relations:
        print("✗ 未能生成任何父子关系")
        return None
    
    # 5. 后处理：移除双重关系
    print(f"\n[步骤5] 后处理：移除双重关系（矛盾的关系）")
    relations = post_process_relations(relations)
    
    if not relations:
        print("✗ 后处理后没有剩余关系")
        return None
    
    # 6. 保存实体关系文件（格式：child,parent）
    print(f"\n[步骤6] 保存父子关系文件: {output_relations_file}")
    os.makedirs(os.path.dirname(output_relations_file) or '.', exist_ok=True)
    with open(output_relations_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['child', 'parent'])
        # 写入关系（格式：child,parent）
        for child, parent in sorted(relations):
            if child and parent:
                writer.writerow([child, parent])
    
    # 统计信息
    parent_stats = {}
    for child, parent in relations:
        if parent not in parent_stats:
            parent_stats[parent] = 0
        parent_stats[parent] += 1
    
    child_stats = {}
    for child, parent in relations:
        if child not in child_stats:
            child_stats[child] = 0
        child_stats[child] += 1
    
    max_children = max(parent_stats.values()) if parent_stats else 0
    avg_children = sum(parent_stats.values()) / len(parent_stats) if parent_stats else 0
    max_parents = max(child_stats.values()) if child_stats else 0
    avg_parents = sum(child_stats.values()) / len(child_stats) if child_stats else 0
    
    print(f"✓ 生成了 {len(relations)} 个父子关系")
    print(f"  父实体统计：{len(parent_stats)} 个父实体，最多 {max_children} 个子实体，平均 {avg_children:.1f} 个子实体")
    print(f"  子实体统计：{len(child_stats)} 个子实体，最多 {max_parents} 个父实体，平均 {avg_parents:.1f} 个父实体")
    print(f"✓ 关系文件已保存到: {output_relations_file}")
    print(f"  格式: child,parent (每行一个关系)")
    
    return output_relations_file


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="从已有的entities和chunks生成父子关系文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从已有的entities和chunks生成父子关系文件（从.env读取配置）
  # 构建方式：
  # 1. 加载已有的实体列表
  # 2. 从chunks中提取共现的实体对（候选对）
  # 3. 使用LLM判断候选对之间的父子关系（child, parent）
  # .env文件内容示例：
  # MODEL_NAME=ge2.5-pro
  # ARK_API_KEY=your_api_key
  # BASE_URL=https://ark.cn-beijing.volces.com/api/v3
  python generate_entities_with_llm.py \\
      --chunks datasets/dart_chunks.txt \\
      --entities entities_new/dart_entities_list.csv \\
      --output-relations entities_new/dart_relations.csv \\
      --batch-size 50 \\
      --max-relations 500
  
  # 手动指定模型（会覆盖.env中的配置）
  python generate_entities_with_llm.py \\
      --chunks datasets/dart_chunks.txt \\
      --entities entities_new/dart_entities_list.csv \\
      --output-relations entities_new/dart_relations.csv \\
      --model gpt-3.5-turbo \\
      --batch-size 50 \\
      --max-relations 500
  
  输出格式：
  - CSV文件，包含表头：child,parent
  - 每行一个父子关系：子实体,父实体
        """
    )
    parser.add_argument('--chunks', type=str, required=True,
                       help='Chunks文件路径（JSON格式或文本格式）')
    parser.add_argument('--entities', type=str, required=True,
                       help='实体列表文件路径（必须提供，CSV格式或文本格式）')
    parser.add_argument('--output-relations', type=str, required=True,
                       help='输出的父子关系文件路径（CSV格式，格式：child,parent）')
    parser.add_argument('--model', type=str, default=None,
                       help='LLM模型名称（默认：从.env文件的MODEL_NAME读取，如果没有则使用gpt-3.5-turbo）')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='每批处理的实体对数量（默认：50）')
    parser.add_argument('--max-relations', type=int, default=500,
                       help='最多生成的关系数（默认：500）')
    parser.add_argument('--use-llm', action='store_true', default=True,
                       help='使用LLM判断父子关系（默认：True，必须启用）')
    parser.add_argument('--no-llm', dest='use_llm', action='store_false',
                       help='不使用LLM（不推荐，无法生成父子关系）')
    parser.add_argument('--start-index', type=int, default=0,
                       help='从指定chunk索引开始处理（默认：0）')
    
    args = parser.parse_args()
    
    # 确保输出文件路径包含.csv扩展名
    output_relations_file = args.output_relations
    if not output_relations_file.endswith('.csv'):
        output_relations_file = output_relations_file + '.csv'
    
    # 如果没有指定模型，从环境变量读取
    model_name = args.model or get_model_name()
    
    relations_file = generate_hierarchical_relations(
        chunks_file=args.chunks,
        entities_file=args.entities,
        output_relations_file=output_relations_file,
        model_name=model_name,
        batch_size=args.batch_size,
        max_relations=args.max_relations,
        use_llm=args.use_llm,
        start_index=args.start_index
    )
    
    print("\n" + "=" * 80)
    if relations_file:
        print("✓ 构建完成！")
        print("=" * 80)
        print(f"\n生成的父子关系文件: {relations_file}")
        print(f"格式: child,parent (每行一个关系)")
    else:
        print("✗ 构建失败")
        print("=" * 80)


if __name__ == "__main__":
    main()
