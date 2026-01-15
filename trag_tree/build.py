# -*- encoding:utf-8 -*-

from entity import ruler
from trag_tree import EntityTree
import csv
import pickle
from trag_tree import hash
import os


def get_dump_file_path1(tree_num_max, entities_file_name, node_num_max):
    safe_name = os.path.basename(entities_file_name)
    return f"./entity_forest_cache/forest_nlp_entities_file_{safe_name}_tree_num_{tree_num_max}_node_num_{node_num_max}_bf1.pkl"

def get_dump_file_path2(tree_num_max, entities_file_name, node_num_max):
    safe_name = os.path.basename(entities_file_name)
    return f"./entity_forest_cache/forest_nlp_entities_file_{safe_name}_tree_num_{tree_num_max}_node_num_{node_num_max}_bf2.pkl"

def get_dump_file_path3(tree_num_max, entities_file_name, node_num_max):
    safe_name = os.path.basename(entities_file_name)
    return f"./entity_forest_cache/forest_nlp_entities_file_{safe_name}_tree_num_{tree_num_max}_node_num_{node_num_max}_bf_cpp.pkl"
    

def build_forest(tree_num_max=30, entities_file_name="entities_file", search_method=1, node_num_max=1000):

    if search_method == 5:
        dump_file_path = get_dump_file_path2(tree_num_max, entities_file_name, node_num_max)
    # elif search_method == 6:
    #     dump_file_path = get_dump_file_path3(tree_num_max, entities_file_name)
    else:
        dump_file_path = get_dump_file_path1(tree_num_max, entities_file_name, node_num_max)

    if search_method != 6 and os.path.exists(dump_file_path):
        print(f"Loading cached forest and nlp from {dump_file_path}...")
        with open(dump_file_path, 'rb') as f:
            forest, nlp = pickle.load(f)

        # Cuckoo filter (search_method == 7) - initialize filter even when loading from cache
        if search_method == 7:
            # Count entities from forest to estimate size
            entities_count = 0
            try:
                for tree in forest:
                    if hasattr(tree, 'all_nodes'):
                        entities_count += len(tree.all_nodes)
            except:
                pass
            # Use a safe default if we can't count
            if entities_count == 0:
                entities_count = 100000
            hash.change_filter(entities_count)
        
        return forest, nlp
    
    rel = []
    entities_list = set()
    with open(entities_file_name+".csv", "r", encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            if len(row) >= 2:
                rel.append({'subject': row[0].strip(), 'object': row[1].strip()})
                entities_list.add(row[0].strip())
                entities_list.add(row[1].strip())

    nlp = ruler.enhance_spacy(list(entities_list))

    # Cuckoo filter (search_method == 7) - initialize filter
    if search_method == 7:
        hash.change_filter(len(entities_list))

    data = []
    root_list = set()
    out_degree = set()
    forest = []

    for dependency in rel:
        data.append([dependency['subject'].lower().strip(), dependency['object'].lower().strip()])
        out_degree.add(dependency['subject'].lower().strip())

    for edge in data:
        if edge[1] not in out_degree:
            root_list.add(edge[1])

    success_num = 0
    count_num = 0
    for root in root_list:
        # print("build tree...")
        new_tree = EntityTree(root, data, search_method)
        
        node_num = new_tree.bfs_count()
        # print(f"tree: {success_num+1}  node_num: {node_num} head: {new_tree.get_root().get_entity()}")
        if node_num+count_num > node_num_max:
            break
        count_num += node_num
        
        forest.append(new_tree)
        success_num += 1
        if success_num > tree_num_max:
            break
    
    print(f"tree num: {success_num}")
    print(f"node num: {count_num}")
    
    if search_method != 6:
        os.makedirs(os.path.dirname(dump_file_path) or '.', exist_ok=True)
        with open(dump_file_path, 'wb') as f:
            pickle.dump((forest, nlp), f)

    return forest, nlp
