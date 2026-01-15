import os
import time
import tiktoken
from typing import Iterable

# 确保在导入OpenAI之前加载.env
from dotenv import load_dotenv
load_dotenv()

from lab_1806_vec_db import VecDB as RagVecDB
# RagMultiVecDB may not exist in this version, use RagVecDB for both
RagMultiVecDB = RagVecDB
from openai import OpenAI

from rag_base.embed_model import get_embed_model
from trag_tree import EntityTree
from entity import ruler
from sentence_transformers import SentenceTransformer, util
import tiktoken

MAX_TOKENS = 16385

# 限制字符串长度
def truncate_to_fit(text, max_tokens, model_name):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except (KeyError, Exception):
        # Fallback to cl100k_base for unknown models (like ge2.5-pro)
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # If tiktoken fails completely, use simple character-based estimation
            # Approximate: 1 token ≈ 4 characters for English, 1.5 for Chinese
            max_chars = max_tokens * 2  # Conservative estimate
            if len(text) <= max_chars:
                return text
            return text[:max_chars] + "..."
    
    tokens = encoding.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return encoding.decode(tokens)


def get_model_name():
    return os.getenv("MODEL_NAME") or "ge2.5-pro"


# 对树节点上下文进行相关度排序
def rank_contexts(query: str, contexts: list, rank_k : int) -> list:
    # 如果contexts为空，直接返回空列表
    if not contexts or len(contexts) == 0:
        return []
    
    model = SentenceTransformer('all-MiniLM-L6-v2')  
    query_embedding = model.encode(query, convert_to_tensor=True)
    context_embeddings = model.encode(contexts, convert_to_tensor=True)
    
    # 如果context_embeddings为空或形状不正确，返回空列表
    if context_embeddings.shape[0] == 0:
        return []
    
    similarities = util.pytorch_cos_sim(query_embedding, context_embeddings)[0]
    
    top_indices = similarities.argsort(descending=True).tolist()
    ranked_contexts = []
    seen_embeddings = []
    
    for idx in top_indices:
        context = contexts[idx]
        context_embedding = context_embeddings[idx]
        
        # 计算与已选上下文的相似度，去除高度相似的项
        if any(util.pytorch_cos_sim(context_embedding, emb) > 0.9 for emb in seen_embeddings):
            continue
        
        ranked_contexts.append(context)
        seen_embeddings.append(context_embedding)
        
        if len(ranked_contexts) == rank_k:
            break
    
    return ranked_contexts


def enrich_results_with_summary_embeddings(results: list, db: RagVecDB | RagMultiVecDB, embed_model, query_embedding: list[float]):
    """
    For each raw_chunk result, find the corresponding tree_node and add its embedding as summary_embedding.
    Two chunks share one abstract (tree_node): chunk_id // 2 gives the pair_id.
    """
    enriched_results = []
    
    # Build maps: pair_id -> tree_node embedding, chunk_id -> chunk content
    tree_node_map = {}
    chunk_content_map = {}
    
    # First pass: collect tree_nodes and chunk contents from search results
    for r in results:
        if r.get("type") == "tree_node":
            pair_id = r.get("pair_id")
            if pair_id is not None:
                tree_text = r.get("content", r.get("title", ""))
                # Use the embedding from the result if available, otherwise compute it
                if "embedding" in r:
                    tree_node_map[pair_id] = r["embedding"]
                else:
                    tree_embedding = embed_model.encode([tree_text], normalize_embeddings=True)[0].tolist()
                    tree_node_map[pair_id] = tree_embedding
        
        elif r.get("type") == "raw_chunk":
            chunk_id = r.get("chunk_id")
            if chunk_id is not None:
                # Convert chunk_id to int for consistent key type
                try:
                    chunk_id_key = int(chunk_id) if isinstance(chunk_id, str) else chunk_id
                    chunk_content_map[chunk_id_key] = r.get("content", r.get("title", ""))
                except (ValueError, TypeError):
                    pass
    
    # Second pass: enrich raw_chunk results with summary_embeddings
    for r in results:
        if r.get("type") == "raw_chunk":
            chunk_id = r.get("chunk_id")
            if chunk_id is not None:
                # Calculate pair_id: two chunks share one abstract (chunk_id // 2)
                # Convert chunk_id to int if it's a string
                try:
                    chunk_id_int = int(chunk_id) if isinstance(chunk_id, str) else chunk_id
                    pair_id = chunk_id_int // 2
                except (ValueError, TypeError):
                    # If conversion fails, skip this result
                    continue
                
                if pair_id in tree_node_map:
                    # Use the existing tree_node embedding
                    r["summary_embedding"] = tree_node_map[pair_id]
                else:
                    # Build the tree_node text from the two chunks that share this abstract
                    chunk_ids_for_pair = [pair_id * 2, pair_id * 2 + 1]
                    merged_text_parts = []
                    for cid in chunk_ids_for_pair:
                        # Try both int and str keys for chunk_content_map
                        if cid in chunk_content_map:
                            merged_text_parts.append(chunk_content_map[cid])
                        elif str(cid) in chunk_content_map:
                            merged_text_parts.append(chunk_content_map[str(cid)])
                    
                    if merged_text_parts:
                        # Merge the two chunks to create the abstract text
                        merged_text = "\n".join(merged_text_parts)
                        # Compute embedding for the merged text (abstract)
                        r["summary_embedding"] = embed_model.encode([merged_text], normalize_embeddings=True)[0].tolist()
                    else:
                        # If we can't build the abstract, use the chunk's own embedding
                        r["summary_embedding"] = r.get("embedding")
                
                enriched_results.append(r)
        
        # For tree_node results, use their own embedding as summary_embedding
        elif r.get("type") == "tree_node":
            if "embedding" in r:
                r["summary_embedding"] = r["embedding"]
            else:
                tree_text = r.get("content", r.get("title", ""))
                r["summary_embedding"] = embed_model.encode([tree_text], normalize_embeddings=True)[0].tolist()
            enriched_results.append(r)
    
    return enriched_results


def filter_contexts_by_dual_threshold(
    results: list,
    query_embedding: list[float],
    threshold_chunk: float = 0.7,
    threshold_summary: float = 0.7,
):
    """
    Keep a chunk iff:
    1) sim(query, chunk) >= threshold_chunk
    2) sim(query, corresponding summary / tree node) >= threshold_summary

    Assumes each result dict contains:
      - "embedding": embedding of the raw chunk
      - "summary_embedding": embedding of the corresponding tree/summary node
    """
    filtered = []
    for r in results:
        # safety check
        if "embedding" not in r or "summary_embedding" not in r:
            continue

        sim_chunk = util.pytorch_cos_sim(
            util.tensor(query_embedding),
            util.tensor(r["embedding"])
        )[0].item()

        sim_summary = util.pytorch_cos_sim(
            util.tensor(query_embedding),
            util.tensor(r["summary_embedding"])
        )[0].item()

        if sim_chunk >= threshold_chunk and sim_summary >= threshold_summary:
            filtered.append(r)

    return filtered


retrieval_time = None
generation_time = None
def get_retrieval_time():
    return retrieval_time
def get_generation_time():
    return generation_time

def augment_prompt(query: str, db: RagVecDB | RagMultiVecDB, forest: list[EntityTree]=None, nlp=None, search_method=1, k=3, model_name="gpt-3.5-turbo", debug=False):
    global retrieval_time
    
    embed_model = get_embed_model()
    
    start_time = time.time()
    input_embedding: list[float] = embed_model.encode([query])[0].tolist()
    
    # Get table_name from db - use module-level dict like build_index.py
    # First try to get from build_index's map (for newly created dbs)
    db_id = id(db)
    table_name = None
    
    # Check build_index module's map
    from rag_base import build_index
    if hasattr(build_index.build_index_on_chunks, '_db_table_map'):
        table_name = build_index.build_index_on_chunks._db_table_map.get(db_id)
    
    # Check load_vec_db module's map
    if table_name is None and hasattr(build_index.load_vec_db, '_db_table_map'):
        table_name = build_index.load_vec_db._db_table_map.get(db_id)
    
    # If still None, get from db directly
    if table_name is None:
        keys = db.get_all_keys()
        table_name = keys[0] if keys else "default_table"
        # Store in load_vec_db's map for future use
        if not hasattr(build_index.load_vec_db, '_db_table_map'):
            build_index.load_vec_db._db_table_map = {}
        build_index.load_vec_db._db_table_map[db_id] = table_name
    
    # VecDB.search requires (key, query, k, ...)
    search_results = db.search(table_name, input_embedding, k * 3)
    # Convert to expected format: list of dicts with "content", "title", "embedding", etc.
    results = []
    for metadata, distance in search_results:
        result = dict(metadata)  # Copy metadata
        result["distance"] = distance
        # Extract embedding if available, otherwise will compute in enrich function
        results.append(result)
    
    # Baseline RAG (search_method=0) 不需要abstract相关的处理
    if search_method == 0 or forest is None:
        # 对于baseline RAG，直接使用向量数据库检索的结果，只取top k
        results = results[:k]
    else:
        # Abstract RAG: Enrich results with summary_embeddings (two chunks share one abstract)
        results = enrich_results_with_summary_embeddings(results, db, embed_model, input_embedding)
        
        # Dual-similarity filtering: query–chunk AND query–summary
        # 保存过滤前的结果，以便在过滤后结果为空时回退
        results_before_filter = results.copy()
        results = filter_contexts_by_dual_threshold(
            results,
            input_embedding,
            threshold_chunk=0.6,
            threshold_summary=0.6,
        )
        
        # 如果过滤后结果为空，回退到未过滤的结果（但只取top k）
        # 这样可以避免因为阈值太严格导致完全没有检索内容
        if len(results) == 0:
            results = results_before_filter[:k]
        else:
            # Keep only top k results after filtering
            results = results[:k]
    end_time = time.time()
    if debug:
        execution_time = end_time - start_time
        # print(f"\n\033[1;31mVectorDB search time: {execution_time:.6f} seconds\033[0m\n")
        # print(f"Search with {k=}, {query=}; Got {len(results)} results")
        # for idx, r in enumerate(results):
        #     title = r["title"]
        #     print(f"{idx}: {title=}")

    source_knowledge = "\n".join([x["content"] for x in results])

    if forest is None or search_method == 0:
        # max_allowed_tokens = MAX_TOKENS - 500
        # source_knowledge = truncate_to_fit(source_knowledge, max_allowed_tokens, model_name)
        # Use English prompt for English datasets, Chinese for Chinese datasets
        # For AESLC (English email summarization), use English prompt
        if any(keyword in query.lower() for keyword in ['summarize', 'email', 'summary']):
            augmented_prompt = (
                f"Use the provided information to answer the question.\n\nInformation:\n{source_knowledge}\n\nQuestion: \n{query}"
            )
        else:
            augmented_prompt = (
                f"使用提供的信息回答问题。\n\n信息:\n{source_knowledge}\n\n问题: \n{query}"
            )
        if debug:
            # print(f"augmented_prompt: {augmented_prompt}")
            pass
        return augmented_prompt

    node_list = []
    
    start_time = time.time()
    
    if search_method in [1, 2, 5, 6]:
        for entity_tree in forest:
            node_list += ruler.search_entity_info(entity_tree, nlp, query, search_method)
    elif search_method == 4:
        node_list = ruler.search_entity_info_naive_hash(nlp, query)
    elif search_method == 7:
        # Enhanced Cuckoo Filter search with hierarchical abstract retrieval
        # This uses the new enhanced function that retrieves abstracts and original text chunks
        try:
            search_context = ruler.search_entity_info_cuckoofilter_enhanced(
                nlp, 
                query,
                db,  # Vector database for retrieving chunks
                embed_model,  # Embedding model
                forest,  # Entity forest for hierarchy traversal
                k=k,  # Number of chunks per abstract
                max_hierarchy_depth=2  # Traverse 1-2 levels up/down
            )
            node_list = [search_context] if search_context else []
        except Exception as e:
            # Fallback to original Cuckoo Filter search if enhanced version fails
            print(f"Warning: Enhanced Cuckoo Filter search failed, falling back to original: {e}")
            search_context = ruler.search_entity_info_cuckoofilter(nlp, query)
            node_list = [search_context] if search_context else []
    elif search_method in [8, 9]:
        node_list = ruler.search_entity_info_ann(nlp, query)
    
    # print(f"\nlength of node_list: {len(node_list)}")    
    
    if search_method != 7:
        # For other search methods, node_list contains EntityNode objects
        node_list = [c.get_context() for c in node_list if c is not None]
    # For search_method == 7, node_list already contains strings from cuckoo filter

    print(f"query entity number: {ruler.entity_number}")

    node_list = rank_contexts(query, node_list, ruler.entity_number)
    
    tree_knowledge = "\n---\n".join(node_list)

    end_time = time.time()
    # tree_knowledge = tree_knowledge[:64000]
    
    max_allowed_tokens = (MAX_TOKENS - 500) // 2
    source_knowledge = truncate_to_fit(source_knowledge, max_allowed_tokens, model_name)
    tree_knowledge = truncate_to_fit(tree_knowledge, max_allowed_tokens, model_name)

    # max_allowed_tokens = (int)((MAX_TOKENS - 500) / 2)
    # source_knowledge = truncate_to_fit(source_knowledge, max_allowed_tokens, model_name)
    # tree_knowledge = truncate_to_fit(tree_knowledge, max_allowed_tokens, model_name)

    # Use English prompt for English queries (e.g., "Summarize the following email")
    if any(keyword in query.lower() for keyword in ['summarize', 'email', 'summary']):
        augmented_prompt = (
            f"Answer the question using the provided information. Be concise and direct.\n\nInformation:\n{source_knowledge}\n\nRelations:\n{tree_knowledge}\n\nQuestion: \n{query}"
        )
    else:
        augmented_prompt = (
            f"请回答问题，可以使用我提供的信息（不保证信息是有用的），在回答中不要有分析我提供信息的内容，直接说答案，答案要简略。\n\n信息:\n{source_knowledge}\n\n关系：\n{tree_knowledge}\n\n问题: \n{query}"
        )

    execution_time = end_time - start_time
    retrieval_time = execution_time

    if debug:
        print(f"\n\033[1;31mEntity retrieval time: {execution_time:.6f} seconds\033[0m\n")
        # print(f"\n\nsource_knowledge: {source_knowledge}")
        # print(f"augmented_prompt: {augmented_prompt}")
    return augmented_prompt

# 使用ARK API配置
api_key = os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY")
base_url = os.environ.get("BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

def rag_complete(
    prompt: str,
    db: RagVecDB | RagMultiVecDB,
    forest: list[EntityTree]=None,
    nlp=None,
    search_method=1,
    model_name: str | None = None,
    debug=False,
) -> Iterable[str]:
    
    global retrieval_time
    global generation_time

    # 使用ARK API配置
    api_key = os.environ.get("ARK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    model_name = model_name or get_model_name()

    start_time = time.time()

    # Check if using ARK API (base_url contains ark.cn-beijing.volces.com)
    is_ark_api = "ark.cn-beijing.volces.com" in base_url
    
    if is_ark_api:
        # ARK API uses responses.create() with input format
        try:
            augmented_content = augment_prompt(prompt, db, forest, nlp, search_method=search_method, model_name=model_name, debug=debug)
            response = client.responses.create(
                model=model_name,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": augmented_content
                            }
                        ]
                    }
                ],
            )
            # ARK API returns complete response, not stream
            # ARK API response format: output is a list
            # [0] = ResponseReasoningItem (reasoning process)
            # [1] = ResponseOutputMessage (actual answer with content)
            text = ""
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
                                    text = content_item.text
                                    break
                                elif isinstance(content_item, dict) and 'text' in content_item:
                                    text = content_item['text']
                                    break
                        # Fallback: check for text attribute directly
                        elif hasattr(item, 'text') and item.text:
                            text = item.text
                            break
            elif hasattr(response, 'choices') and response.choices:
                # Standard OpenAI format fallback
                text = response.choices[0].message.content if hasattr(response.choices[0].message, 'content') else str(response.choices[0])
            elif hasattr(response, 'content'):
                text = response.content
            
            # If still no text, try output_text attribute
            if not text and hasattr(response, 'output_text'):
                text = response.output_text
            
            # Last resort: convert to string
            if not text:
                text = str(response)
            
            # Yield text in chunks to maintain compatibility
            if text and len(text) > 0:
                chunk_size = 10
                for i in range(0, len(text), chunk_size):
                    yield text[i:i+chunk_size]
        except AttributeError:
            # Fallback to chat.completions if responses.create doesn't exist
            is_ark_api = False
    
    if not is_ark_api:
        # Standard OpenAI API format
        stream = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. You should use the same language as the user.",
                },
                {
                    "role": "user",
                    "content": augment_prompt(prompt, db, forest, nlp, search_method=search_method, model_name=model_name, debug=debug),
                },
            ],
            stream=True,
        )
        for chunk in stream:
            choices = chunk.choices
            if len(choices) == 0:
                break
            content = choices[0].delta.content
            if content is not None:
                yield content

    end_time = time.time()

    generation_time = end_time - start_time

    # 只有当retrieval_time不为None时才计算Time Ratio
    if retrieval_time is not None and (retrieval_time + generation_time) > 0:
        print(f"\n\033[1;31mTime Ratio: {(retrieval_time*100.0)/(retrieval_time+generation_time):.6f}%\033[0m")
    else:
        print(f"\n\033[1;31mGeneration Time: {generation_time:.6f} seconds\033[0m")
