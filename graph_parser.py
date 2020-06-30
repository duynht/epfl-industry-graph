import pickle
import json
import os
from glob import glob

def get_node_id(node_dict, key):
    if not (key in node_dict):
        global node_counter #self
        node_counter += 1 
        node_dict[key] = node_counter
    
    return node_dict[key]

if __name__ == '__main__':
    
    num_entries = 10000
    node_counter = -1
    node_dict = {}
    node_type_dict = {}
    edge_list = []
    raw_path = 'data/raw/patent-extracted/using/**'
    for filepath in sorted(glob(raw_path, recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        with open(filepath,encoding='utf-8') as f:
            for line_num, line in enumerate(f):       
                if not line:
                    break

                if line_num == num_entries:
                    break

                data = json.loads(line)
                
                company_node = get_node_id(node_dict, data['company_name'])
                node_type_dict[company_node] = 'company'

                for field in data['fields']:
                    field_node = get_node_id(node_dict, field)
                    edge_list.append((company_node, field_node)) # already assumed undirected
                    node_type_dict[field_node] = 'field'

    with open('data/parsed-graph/graph.txt','w') as f:
        for edge in edge_list:
            f.write('{0} {1}\n'.format(edge[0], edge[1]))

    with open('data/parsed-graph/str2id_dict.pkl', 'wb') as f:
        pickle.dump(node_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    node_dict = {value : key for key, value in node_dict.items()}
    with open('data/parsed-graph/node_dict.pkl','wb') as f:
        pickle.dump(node_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open('data/parsed-graph/node_type_dict.pkl','wb') as f:
        pickle.dump(node_type_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

