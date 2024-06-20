import xml.etree.ElementTree as ET
import re
from config import *
from collections import defaultdict

tree = ET.parse(f"{MODEL_DIRECTORY}/openvino_model.xml")
root = tree.getroot()

def parse_layers(string):
    match1 = re.search(r'\/model\/layers\.(\d+)\/(\w+)', string)
    if match1:
        layer_number = int(match1.group(1))
        phrase = match1.group(2)
        return layer_number, phrase

    return -1, ""

topology = {
    "initial": defaultdict(set),
    "transformer_layers": [],
}

def profile_set(set: defaultdict[set]) -> defaultdict[int]:
    defaultdict_int = defaultdict(int)
    for key, value_set in set.items():
        defaultdict_int[key] = len(value_set)
    return dict(defaultdict_int)

def profile_topology(t):
    profiled_topology = {}
    profiled_topology["initial"] = profile_set(t["initial"])
    profiled_topology["transformer_layers"] = []
    for set_ in t["transformer_layers"]:
        profiled_topology["transformer_layers"].append(profile_set(set_))
    return profiled_topology

current_layer = -1
start_layer_index = -1

operation_types = set()

# For verification
layers_added = 0
total_iterations = 0
transformer_list = topology["transformer_layers"]

# Caching purposes
idx_to_name = [] # layer id in xml -> name
name_to_idx = {}

# ======= [PART 1] Topology creation =======

for layer in sorted(root.findall('.//layer'), key=lambda x: int(x.get('id'))):
    id, name, type = int(layer.get('id')), layer.get('name'), layer.get('type')
    transformer_block_num, group = parse_layers(name) # phrase
    if transformer_block_num == -1:
        group = type
    if group == "Add_1": # Hypothesis: all Add_1 indicates the end of a transformer block
        transformer_block_num += 1

    # See if we are entering a new transformer block
    if transformer_block_num > -1:
        if current_layer != transformer_block_num:
            print(f"\n >>> BLOCK: {transformer_block_num}, ID: {id} <<< \n")
            current_layer = transformer_block_num
            start_layer_index = id
            
            assert len(transformer_list) == current_layer
            transformer_list.append(defaultdict(set))
            transformer_list[-1]["Other"] = defaultdict(set)
        
        if group == "Add_1": # Hypothesis: all Add_1 indicates the end of a transformer block
            transformer_list[current_layer-1][group].add(id)
        else:
            transformer_list[current_layer][group].add(id)
        layers_added += 1
    
    elif transformer_block_num == -1 and current_layer > -1:
        transformer_list[current_layer]["Other"][group].add(id)
        layers_added += 1
        pass

    elif current_layer == -1:
        topology["initial"][group].add(id)
        layers_added += 1
    total_iterations += 1
    
    # Add to dictionary lookup
    idx_to_name.append(name)
    name_to_idx[name] = id

    # stdout Profiling information - printing information
    if id % 100 == 0 or start_layer_index <= id <= start_layer_index + 3:
        print(f'ID: {id}, Name: {name}')

print(f"Layers added: {layers_added} / {total_iterations}")

for i in range(len(idx_to_name)):
    assert i == name_to_idx[idx_to_name[i]]

# ======= [PART 2] Partitioning QKV Cache =======

# Some scouting stuff
start_qs = []
for i in range(0, 32): # transformer blocks 0-31
    print(i)
    q = name_to_idx[f"/model/layers.{i}/self_attn/q_proj/MatMul"]
    k = name_to_idx[f"/model/layers.{i}/self_attn/k_proj/MatMul"]
    v = name_to_idx[f"/model/layers.{i}/self_attn/v_proj/MatMul"]
    print(v-k, k-q)
    start_qs.append(q)
    pass

for i in range(1, len(start_qs)):
    print(start_qs[i] - start_qs[i-1])


# ============ OUTPUT CONFIGS ============

# from pprint import pprint

# # Profiling: convert each list into a number
# for i, profile in enumerate(transformer_list):
#     profiled = profile_set(profile)
#     profiled["Other"] = profile_set(profile["Other"])
#     print(i)
#     pprint(profiled)

