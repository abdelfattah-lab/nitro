from config import *
import xml.etree.ElementTree as ET
import re

def create_layer_mappings(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Initialize dictionaries
    idx_to_name = []
    idx_to_type= []
    name_to_idx = {}
    
    # Find all layers
    layers = root.findall('.//layer')
    layers = sorted(layers, key=lambda x: int(x.get('id')))
    
    # Iterate through each layer
    for layer in layers:
        layer_id = layer.get('id')
        layer_name = layer.get('name')
        layer_type = layer.get('type')
        
        # Save to idx_to_name list
        idx_to_name.append(layer_name)
        idx_to_type.append(layer_type)

        # Save to name_to_idx dictionary
        name_to_idx[layer_name] = int(layer_id)
    
    return idx_to_name, name_to_idx, idx_to_type

# Function to delete a layer by id
def delete_layer_by_id(file_path, layer_id, output_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find the layers element
    layers = root.find('layers')
    
    if layers is not None:
        # Find the layer with the specified id
        layer_to_delete = layers.find(f"./layer[@id='{layer_id}']")
        
        if layer_to_delete is not None:
            # Remove the layer
            layers.remove(layer_to_delete)
            print(f"Layer with id {layer_id} has been deleted.")
        else:
            print(f"Layer with id {layer_id} not found.")
            
    else:
        print("No layers element found in the XML.")
    
    # Write the modified XML to a new file
    xml_declaration = '<?xml version="1.0"?>\n'
    xml_content = ET.tostring(root, encoding='unicode')
    
    with open(output_path, 'w') as f:
        f.write(xml_declaration)
        f.write(xml_content)

def find_past_key_values_ids(file_path):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Regular expression patterns
    pattern_value = re.compile(r"^past_key_values\.(\d+)\.value$")
    pattern_key = re.compile(r"^past_key_values\.(\d+)\.key$")
    
    # List to save matching IDs
    matching_ids = []
    
    # Find the layers element
    layers = root.findall('.//layer')
    
    # Iterate through each layer
    for layer in layers:
        layer_name = layer.get('name')
        if layer_name:
            match_value = pattern_value.match(layer_name)
            match_key = pattern_key.match(layer_name)
            if match_value:
                X = int(match_value.group(1))
                if 0 <= X <= 31:
                    layer_id = layer.get('id')
                    matching_ids.append(layer_id)
            # elif match_key:
            #     X = int(match_key.group(1))
            #     if 0 <= X <= 31:
            #         layer_id = layer.get('id')
            #         matching_ids.append(layer_id)
    
    return matching_ids

def find_input_nodes(file_path, layer_ids):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find the edges element
    edges = root.findall('.//edge')
    
    # Dictionary to save connections
    connections = {layer_id: [] for layer_id in layer_ids}
    
    # Iterate through each edge
    for edge in edges:
        from_layer = edge.get('to-layer')
        to_layer = edge.get('from-layer')
        
        if from_layer in layer_ids:
            connections[from_layer].append(to_layer)
    
    for layer in connections:
        connections[layer].sort(reverse=True)
    
    return connections

def find_output_nodes(file_path, layer_ids):
    # Parse the XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Find the edges element
    edges = root.findall('.//edge')
    
    # Dictionary to save connections
    connections = {layer_id: [] for layer_id in layer_ids}
    
    # Iterate through each edge
    for edge in edges:
        from_layer = edge.get('from-layer')
        to_layer = edge.get('to-layer')
        
        if from_layer in layer_ids:
            connections[from_layer].append(to_layer)
    
    for layer in connections:
        connections[layer].sort(reverse=True)
    
    return connections

def create_new_edges(input_file, mapping, output_file):
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Find the <edges> element to append new <edge> elements
    edges_elem = root.find('edges')
    if edges_elem is None:
        raise ValueError("Could not find <edges> element in the XML.")
    
    count = 0
    # Iterate through the mapping dictionary
    for from_layer_id, to_layers in mapping.items():
        # Create a new <edge> element for each to-layer
        for to_layer_id in to_layers:
            # Create <edge> element
            edge_elem = ET.Element('edge')
            
            # Set attributes
            edge_elem.set('from-layer', str(from_layer_id))
            edge_elem.set('from-port', '2')  # Assuming from-port is always 2
            edge_elem.set('to-layer', str(to_layer_id))
            edge_elem.set('to-port', '0')  # Assuming to-port is always 0
            count += 1
            
            # Append <edge> element to <edges> section
            edges_elem.append(edge_elem)
    
    print(f"Edges added: {count}")
    # Write the modified XML back to the file
    tree.write(output_file, encoding='utf-8', xml_declaration=True, method="xml",  short_empty_elements=False)

def remove_nodes_and_edges(input_file, matching_ids, output_file):
    # Parse the XML file
    tree = ET.parse(input_file)
    root = tree.getroot()
    
    # Find all layers and edges
    layers_elem = root.find('layers')
    edges_elem = root.find('edges')
    
    if layers_elem is None or edges_elem is None:
        raise ValueError("Could not find <layers> or <edges> element in the XML.")
    
    # Collect nodes and edges to remove
    nodes_to_remove = set()
    edges_to_remove = []
    
    for layer in layers_elem.findall('layer'):
        layer_id = layer.get('id')
        
        if layer_id in matching_ids:
            nodes_to_remove.add(layer)
    
    for edge in edges_elem.findall('edge'):
        from_layer = edge.get('from-layer')
        to_layer = edge.get('to-layer')
        
        if from_layer in matching_ids or to_layer in matching_ids:
            edges_to_remove.append(edge)
    
    # Remove nodes and edges
    for node in nodes_to_remove:
        layers_elem.remove(node)
    
    print(f"Nodes removed: {len(nodes_to_remove)}")
    
    for edge in edges_to_remove:
        edges_elem.remove(edge)
    
    print(f"Edges removed: {len(edges_to_remove)}")
    
    # Write the modified XML to a new file
    tree.write(output_file, encoding='utf-8', xml_declaration=True)

# DRIVER
input_file = f"{MODEL_DIRECTORY}/openvino_model.xml"
output_file = f"{MODEL_DIRECTORY}/openvino_model_modified.xml"

# Creating id-mappings
idx_to_name, name_to_idx, idx_to_type = create_layer_mappings(input_file)
for i in range(len(idx_to_name)):
    assert i == name_to_idx[idx_to_name[i]]

# Obtain all [past_key_values.X.value] operations
matching_ids = find_past_key_values_ids(input_file)
# matching_ids.remove("63")
print(f"Matching layer IDs: {matching_ids}")

# Obtain all concat operations connected to the above parameter
concat_nodes = find_output_nodes(input_file, matching_ids)
concat_nodes = [concat_nodes[key][0] for key in concat_nodes]
print(f"Connected nodes: {concat_nodes}")

# Obtain all input nodes to the concat -- should be one PARAMETER and one TRANSPOSE
input_nodes = find_input_nodes(input_file, concat_nodes)
purger = []
for concat_node in input_nodes: # Prune this list to contain only concat node
    lst = input_nodes[concat_node]
    if len(lst) == 1:
        purger.append(concat_node)
        continue
    input_nodes[concat_node] = lst[0] if idx_to_type[int(lst[0])] == "Transpose" or idx_to_type[int(lst[0])] == "Add_1" else lst[1]
for node in purger:
    input_nodes.pop(node)
    concat_nodes.remove(node)

print(f"Input nodes: {input_nodes}")

# Obtain unsqueeze, shape, sink_port from the concat operation
output_nodes = find_output_nodes(input_file, concat_nodes)
sink_nodes = []
for node in output_nodes:
    outputs = output_nodes[node]
    print(outputs)
    for o in outputs:
        if idx_to_name[int(o)].endswith("sink_port_0"):
            sink_nodes.append(o)
    outputs.remove(sink_nodes[-1])

print(f"Output nodes: {output_nodes}")

# New mapping:
mapping = {}
for concat in input_nodes:
    assert concat in output_nodes
    mapping[int(input_nodes[concat])] = []
    for output in output_nodes[concat]:
        mapping[int(input_nodes[concat])].append(int(output))
print(mapping)

create_new_edges(input_file, mapping, output_file)
remove_nodes_and_edges(output_file, matching_ids + sink_nodes + concat_nodes, output_file)
print("New edges have been added to the XML file.")