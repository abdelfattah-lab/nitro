import xml.etree.ElementTree as ET
from xml.dom import minidom

import shutil
import os

def parse_and_rename_layers(file, output_file=None):
    if output_file is None:
        output_file = file
    tree = ET.parse(file)
    root = tree.getroot()

    # Find all layers
    layers = root.findall('.//layer')
    
    # Find all edges
    edges = root.findall('.//edge')
    
    for layer in layers:
        layer_id = layer.get('id')
        layer_name = layer.get('name')
        layer_type = layer.get('type')
        
        # Check if the layer type is "Result"
        if layer_type == "Result":
            input_ports = layer.findall('.//input/port')
            for port in input_ports:
                dims = port.findall('.//dim')
                dim_values = [int(dim.text) for dim in dims]
                
                # Check for cache_v or cache_k
                if len(dim_values) == 4:
                    for edge in edges:
                        if edge.get('to-layer') == layer_id:
                            from_layer_id = edge.get('from-layer')
                            from_layer = root.find(f".//layer[@id='{from_layer_id}']")
                            if from_layer is not None:
                                from_layer_name = from_layer.get('name')
                                module_index = from_layer_name.split('.')[2].split('/')[0]
                                output_ports = from_layer.findall('.//output/port')
                                for port in output_ports:
                                    port_name = port.get('names')
                                    if port_name is not None:
                                        if port_name.startswith('cache_v'):
                                            new_name = f'cache_v_{module_index}_out'
                                        elif port_name.startswith('cache_k'):
                                            new_name = f'cache_k_{module_index}_out'
                                        else:
                                            continue
                                        layer.set('name', new_name)
                                        port.set('names', new_name)
                                        break
                            else:
                                layer.set('name', 'logit')
                
                # Check for logit
                elif len(dim_values) == 3:
                    layer.set('name', 'logit')
                    # Update output port names in the "from-layer"
                    for edge in edges:
                        if edge.get('to-layer') == layer_id:
                            from_layer_id = edge.get('from-layer')
                            from_layer = root.find(f".//layer[@id='{from_layer_id}']")
                            if from_layer is not None:
                                output_ports = from_layer.findall('.//output/port')
                                for port in output_ports:
                                    port_name = port.get('names')
                                    if port_name is not None:
                                        # Rename the output port to "logits"
                                        port.set('names', 'logits')
                                        break
    
    # Save the modified XML
    tree.write(output_file)

    # Copy the BIN, if different
    if output_file != file:
        input_bin_file = os.path.splitext(file)[0] + '.bin'
        output_bin_file = os.path.splitext(output_file)[0] + '.bin'
        shutil.copyfile(input_bin_file, output_bin_file)


def add_rv_inputs(file, output_file=None):
    if output_file is None:
        output_file = file

    try:
        # Load the XML file
        tree = ET.parse(file)
        root = tree.getroot()

        # Flag to check if changes were made
        changes_made = False

        # Iterate through all <layer> elements
        for layer in root.findall('.//layer'):
            # Check if the layer type is "ReadValue"
            if layer.get('type') == 'ReadValue':
                print(f'Processing layer: {layer.get("name")}')
                
                # Create an <input> element with the same structure as described
                input_element = ET.Element('input')
                port_element = ET.SubElement(input_element, 'port', id='0', precision='FP32')
                
                # Copy the dimensions from the output element
                output = layer.find('output')
                if output is not None:
                    port = output.find('port')
                    if port is not None:
                        for dim in port.findall('dim'):
                            ET.SubElement(port_element, 'dim').text = dim.text
                        
                        # Find the index of the <output> element
                        output_index = list(layer).index(output)
                        
                        # Insert the new <input> element before the <output> element
                        layer.insert(output_index, input_element)
                        changes_made = True
                else:
                    print(f'No output found in layer: {layer.get("name")}')
        
        if changes_made:
            # Save the modified XML to the specified output file
            tree.write(output_file, encoding='utf-8', xml_declaration=True)
            print(f'Changes saved to {output_file}')
        else:
            print('No changes were made.')

    except Exception as e:
        print(f'An error occurred: {e}')