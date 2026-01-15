import json
import csv
import spacy
import networkx as nx
from tqdm import tqdm
from collections import Counter

# 加载模型
nlp = spacy.load("en_core_web_sm")

# 停用词扩展：过滤掉太通用的词
custom_stopwords = {"it", "which", "they", "this", "that", "there", "who", "whom", "whose"}

def get_main_phrase(token, doc):
    """
    获取 token 所在的完整名词短语。如果不在短语中，则返回其文本。
    """
    for chunk in doc.noun_chunks:
        if token.i >= chunk.start and token.i < chunk.end:
            # 过滤掉冠词 (the, a, an)
            return chunk.root.text if len(chunk.text) < 3 else chunk.text
    return token.text

def extract_broad_relations(text):
    doc = nlp(text)
    relations = []
    
    for sent in doc.sents:
        for token in sent:
            # 支持 动词 (VERB) 和 系动词 (AUX，如 is/are)
            if token.pos_ in ("VERB", "AUX"):
                # 提取主语和宾语
                subjects = [w for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                objects = [w for w in token.rights if w.dep_ in ("dobj", "pobj", "attr", "acomp")]
                
                for s in subjects:
                    # 过滤代词和停用词
                    if s.pos_ == "PRON" or s.text.lower() in custom_stopwords:
                        continue
                        
                    for o in objects:
                        if o.pos_ == "PRON" or o.text.lower() in custom_stopwords:
                            continue
                        
                        # 获取更完整的名词短语
                        s_phrase = get_main_phrase(s, doc)
                        o_phrase = get_main_phrase(o, doc)
                        
                        # 清洗格式
                        s_clean = " ".join(s_phrase.split()).strip()
                        o_clean = " ".join(o_phrase.split()).strip()
                        
                        # 基本长度过滤
                        if len(s_clean) > 2 and len(o_clean) > 2 and s_clean.lower() != o_clean.lower():
                            relations.append((s_clean, o_clean))
                            
    return relations

def paper_style_filter(relations):
    """ 保持论文的 4 种过滤规则 """
    # 计数以保留高频关系（可选：如果关系依然太多，可以开启）
    # unique_rels = [rel for rel, count in Counter(relations).items() if count >= 1]
    
    G = nx.DiGraph()
    G.add_edges_from(set(relations))
    
    # 1. 移除环路
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G)
            G.remove_edge(cycle[-1][0], cycle[-1][1])
        except: break
            
    # 2. 传递性规约 (Transitive Reduction)
    try:
        TR = nx.transitive_reduction(G)
        return list(TR.edges())
    except:
        return list(G.edges())

def main(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    all_raw_relations = []
    print(f"正在处理 {len(chunks)} 条 Chunk...")
    
    # 为了提速，可以使用 nlp.pipe
    for text in tqdm(chunks):
        if isinstance(text, str) and len(text) > 10:
            # 限制长度防止处理超长文本卡顿
            all_raw_relations.extend(extract_broad_relations(text[:1500]))
    
    print(f"原始关系总数: {len(all_raw_relations)}")
    
    # 过滤
    final_relations = paper_style_filter(all_raw_relations)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject', 'object'])
        writer.writerows(final_relations)
    
    print(f"完成！生成了 {len(final_relations)} 条关系。")

if __name__ == "__main__":
    # 请确保路径正确
    main('./datasets/aalcr_chunks.json', './entities_new/aalcr_relations_sm.csv')