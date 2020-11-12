import argparse
import pickle
import json
import os
from glob import glob
import csv

def get_register_dict(datapath):        
    register_path = os.path.join(datapath, 'truth/register/*')

    register_dict = defaultdict(lambda: 'Unknown')

    for filepath in sorted(glob(register_path, recursive=True)):
        if os.path.isdir(filepath): continue
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                register_dict[data['address']['organisation'].lower()] = data['uid'] 

    return register_dict

def get_node_id(node_counter, node_dict, key):
    if not (key in node_dict):
        # global node_counter #self
        node_counter += 1 
        node_dict[key] = node_counter
    
    return node_counter, node_dict[key]

def parse_graph(datapath, num_nodes):
    node_counter = -1
    node_dict = {}
    node_type_dict = {}
    edge_list = []
    extracted_path = os.path.join(datapath, 'extracted/**')

    for filepath in sorted(glob(extracted_path, recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        with open(filepath,encoding='utf-8') as f:
            for line in f:       

                data = json.loads(line)
                if data['fields']:
                    node_counter, company_node = get_node_id(node_counter, node_dict, data['company_name'])
                    node_type_dict[company_node] = 'company'

                    for field in data['fields']:
                        node_counter, field_node = get_node_id(node_counter, node_dict, field)
                        edge_list.append((company_node, field_node)) # already assumed undirected
                        node_type_dict[field_node] = 'field'
                    
                    if num_nodes and node_counter + 2 >= num_nodes:
                        break
        if num_nodes and node_counter + 2 >= num_nodes:
            break

    print(node_counter + 2)

    # with open('../data/parsed-graph/_graph.txt','w') as f:
    #     for edge in edge_list:
    #         f.write('{0} {1}\n'.format(edge[0], edge[1]))

    output_dir = os.path.join(datapath,'parsed-graph')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, 'pt_graph.csv')
    with open(filepath,'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id1','id2'])
        for edge in edge_list:
            csv_writer.writerow(edge)

    filepath = os.path.join(output_dir, 'pt_str2id_dict.pkl')
    with open(filepath, 'wb') as f:
        pickle.dump(node_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    node_dict = {value : key for key, value in node_dict.items()}
    filepath = os.path.join(output_dir, 'pt_node_dict.pkl')
    with open(filepath,'wb') as f:
        pickle.dump(node_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    filepath = os.path.join(output_dir, 'pt_node_type_dict.pkl')
    with open(filepath,'wb') as f:
        pickle.dump(node_type_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_nodes', 
                        default=None,
                        type=int,
                        help='number of nodes to be parsed (defaulted to parse all)')
    
    parser.add_argument('-d', '--datapath', 
                        default='../data',
                        type=str,
                        help='path to data directory (default is "../data")')

    args = parser.parse_args()

    parse_graph(args.datapath, args.num_nodes)
    print('Done parsing!')
