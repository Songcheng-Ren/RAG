import os


from lab_1806_vec_db import VecDB as RagVecDB
from tqdm.autonotebook import tqdm

from rag_base.embed_model import get_embed_model

from nltk.tokenize import sent_tokenize
# Tree and Node are not used in this file, remove unused imports
# from trag_tree.tree import Tree
# from trag_tree.node import Node


def split_string_by_headings(text: str):
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
    """从文件收集chunks，支持.txt和.json文件"""
    # 检查文件扩展名
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        # 处理JSON文件（如DART数据集）
        import json
        chunks = []
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                # 尝试加载为JSON数组
                try:
                    data = json.load(file)
                    if isinstance(data, list):
                        # 如果是数组，提取每个元素的文本内容
                        for item in data:
                            if isinstance(item, dict):
                                # 首先检查annotations字段（DART数据集格式）
                                if 'annotations' in item and isinstance(item['annotations'], list) and len(item['annotations']) > 0:
                                    ann = item['annotations'][0]
                                    if isinstance(ann, dict) and 'text' in ann:
                                        chunks.append(str(ann['text']).strip())
                                        continue
                                
                                # 然后检查其他文本字段
                                text_fields = ['answer', 'text', 'target', 'expected_answer']
                                for field in text_fields:
                                    if field in item:
                                        text = item[field]
                                        if isinstance(text, str) and text.strip():
                                            chunks.append(text.strip())
                                            break
                    else:
                        # 如果是单个对象，尝试提取文本
                        if isinstance(data, dict):
                            for field in ['answer', 'text', 'target', 'expected_answer']:
                                if field in data:
                                    chunks.append(str(data[field]).strip())
                                    break
                except json.JSONDecodeError:
                    # 如果不是标准JSON，尝试按JSONL处理（每行一个JSON对象）
                    file.seek(0)
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            if isinstance(item, dict):
                                for field in ['answer', 'text', 'target', 'expected_answer']:
                                    if field in item:
                                        chunks.append(str(item[field]).strip())
                                        break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"Warning: Failed to process JSON file {file_path}: {e}")
            return []
        
        # 如果没有提取到chunks，返回空列表
        if not chunks:
            print(f"Warning: No text content extracted from {file_path}")
        return chunks
    else:
        # 处理文本文件
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read()
            return split_string_by_headings(data)


def collect_chunks_from_dir(dir: str):
    chunks: list[str] = []
    for filename in os.listdir(dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(dir, filename)
            chunks.extend(collect_chunks_from_file(file_path))
    return chunks


def collect_chunks(dir_or_file: str):
    if os.path.isdir(dir_or_file):
        return collect_chunks_from_dir(dir_or_file)
    return collect_chunks_from_file(dir_or_file)


def expand_chunks_to_tree_nodes(chunks: list[str]):
    """
    Build a mixed knowledge base:
    - Keep every raw chunk as a raw knowledge node
    - Every two consecutive chunks share ONE summary (tree node)

    This forms a coarser hierarchical structure suitable for document-level abstraction.
    """
    items = []

    # 1) always keep raw chunks
    for idx, chunk in enumerate(chunks):
        items.append({
            "text": chunk,
            "meta": {
                "type": "raw_chunk",
                "chunk_id": idx,
            }
        })

    # 2) build one summary node for every two chunks
    pair_id = 0
    for i in range(0, len(chunks), 2):
        merged_text = chunks[i]
        related_chunk_ids = [i]

        if i + 1 < len(chunks):
            merged_text = merged_text + "\n" + chunks[i + 1]
            related_chunk_ids.append(i + 1)

        items.append({
            "text": merged_text,
            "meta": {
                "type": "tree_node",
                "pair_id": pair_id,
                "chunk_ids": related_chunk_ids,
                "covered_chunks": related_chunk_ids,
            }
        })
        pair_id += 1

    return items


def build_index_on_chunks(chunks: list[str], batch_size: int = 100, target_dir: str = None):
    items = expand_chunks_to_tree_nodes(chunks)
    batch_size = 64
    model = get_embed_model()
    dim = model.get_sentence_embedding_dimension()
    assert isinstance(dim, int), "Cannot get embedding dimension"

    # VecDB requires a directory path as first argument
    import tempfile
    import os
    # 如果提供了目标目录，直接使用；否则使用临时目录
    if target_dir:
        db_dir = target_dir
    else:
        # 使用进程ID确保临时目录唯一，并与load_vec_db的查找逻辑匹配
        db_dir = os.path.join(tempfile.gettempdir(), f"vec_db_temp_{os.getpid()}")
    os.makedirs(db_dir, exist_ok=True)
    db = RagVecDB(db_dir)
    # Create table with dimension
    table_name = "default_table"
    db.create_table_if_not_exists(table_name, dim)

    for i in tqdm(range(0, len(items), batch_size)):
        i_end = min(len(items), i + batch_size)
        batch = items[i:i_end]
        texts = [it["text"] for it in batch]
        vecs = model.encode(texts, normalize_embeddings=True)
        # batch_add requires key, vec_list, metadata_list
        vec_list = vecs.tolist()
        metadata_list = [
            {
                "content": it["text"][:1000] if len(it["text"]) > 1000 else it["text"],  # 限制长度
                "title": it["text"][:200] if len(it["text"]) > 200 else it["text"],  # 限制长度
                **{k: str(v) for k, v in it["meta"].items()},  # 确保所有值都是字符串
            }
            for it in batch
        ]
        db.batch_add(table_name, vec_list, metadata_list)
    
    # Store table_name - use a wrapper or store in a global dict
    # Since VecDB doesn't allow setting attributes, we'll use a module-level dict
    if not hasattr(build_index_on_chunks, '_db_table_map'):
        build_index_on_chunks._db_table_map = {}
    build_index_on_chunks._db_table_map[id(db)] = table_name

    return db


vec_db_cache_dir = "vec_db_cache/"


def ensure_vec_db_cache_dir():
    if not os.path.exists(vec_db_cache_dir):
        os.mkdir(vec_db_cache_dir)


def cache_path_for_key(key: str):
    ensure_vec_db_cache_dir()
    return os.path.join(vec_db_cache_dir, f"{key}.db")


def load_vec_db(key: str, dir_or_file: str):
    index_path = cache_path_for_key(key)
    if os.path.exists(index_path) and os.path.isdir(index_path):
        # VecDB uses directory path directly
        db = RagVecDB(index_path)
        # Store table_name in module-level dict
        if not hasattr(load_vec_db, '_db_table_map'):
            load_vec_db._db_table_map = {}
        keys = db.get_all_keys()
        if keys:
            load_vec_db._db_table_map[id(db)] = keys[0]
            return db
        else:
            # Existing cache directory but no tables; rebuild from source
            try:
                import shutil
                shutil.rmtree(index_path, ignore_errors=True)
                os.makedirs(index_path, exist_ok=True)
                chunks = collect_chunks(dir_or_file)
                db = build_index_on_chunks(chunks, target_dir=index_path)
                try:
                    db.force_save()
                except:
                    pass
                # Record table name
                if hasattr(build_index_on_chunks, '_db_table_map'):
                    table_name = build_index_on_chunks._db_table_map.get(id(db), "default_table")
                else:
                    ks = db.get_all_keys()
                    table_name = ks[0] if ks else "default_table"
                load_vec_db._db_table_map[id(db)] = table_name
                return db
            except Exception:
                # Fallback to default behavior; search will try to resolve keys later
                load_vec_db._db_table_map[id(db)] = "default_table"
                return db

    # Build new database - 直接在目标位置构建，避免移动文件的问题
    chunks = collect_chunks(dir_or_file)
    
    # 确保目标目录不存在或为空
    if os.path.exists(index_path):
        import shutil
        shutil.rmtree(index_path, ignore_errors=True)
    os.makedirs(index_path, exist_ok=True)
    
    # 直接在目标位置构建数据库
    db = build_index_on_chunks(chunks, target_dir=index_path)
    
    # 确保保存完成
    try:
        db.force_save()
    except:
        pass
    
    # Get table_name from the map
    db_id = id(db)
    if hasattr(build_index_on_chunks, '_db_table_map'):
        table_name = build_index_on_chunks._db_table_map.get(db_id, "default_table")
        # Store table_name for the db instance
        if not hasattr(load_vec_db, '_db_table_map'):
            load_vec_db._db_table_map = {}
        load_vec_db._db_table_map[id(db)] = table_name
    else:
        # 如果没有map，尝试从db获取
        keys = db.get_all_keys()
        table_name = keys[0] if keys else "default_table"
        if not hasattr(load_vec_db, '_db_table_map'):
            load_vec_db._db_table_map = {}
        load_vec_db._db_table_map[id(db)] = table_name
    
    return db
