import pickle
import json
import os
from glob import glob
import csv


def get_node_id(node_dict, key):
    if not (key in node_dict):
        global node_counter #self
        node_counter += 1 
        node_dict[key] = node_counter
    
    return node_dict[key]

if __name__ == '__main__':
    
    num_entries = 1000
    node_counter = -1
    node_dict = {}
    node_type_dict = {}
    edge_list = []
    raw_path = '../data/extracted/**'
    for filepath in sorted(glob(raw_path, recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        with open(filepath,encoding='utf-8') as f:
            for line in f:       
                # if not line:
                #     break

                # if line_num == num_entries:
                #     break

                data = json.loads(line)
                if data['fields']:
                    company_node = get_node_id(node_dict, data['company_name'])
                    node_type_dict[company_node] = 'company'

                    for field in data['fields']:
                        field_node = get_node_id(node_dict, field)
                        edge_list.append((company_node, field_node)) # already assumed undirected
                        node_type_dict[field_node] = 'field'
                    
                    # REMOVE THIS
                    if node_counter + 2 >= num_entries:
                        break
        if node_counter + 2 >= num_entries:
            break

    print(node_counter + 2)

    # with open('../data/parsed-graph/_graph.txt','w') as f:
    #     for edge in edge_list:
    #         f.write('{0} {1}\n'.format(edge[0], edge[1]))
    
    with open('../data/parsed-graph/pt_graph.csv','w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id1','id2'])
        for edge in edge_list:
            csv_writer.writerow(edge)

    with open('../data/parsed-graph/pt_str2id_dict.pkl', 'wb') as f:
        pickle.dump(node_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    node_dict = {value : key for key, value in node_dict.items()}
    with open('../data/parsed-graph/pt_node_dict.pkl','wb') as f:
        pickle.dump(node_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../data/parsed-graph/pt_node_type_dict.pkl','wb') as f:
        pickle.dump(node_type_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

