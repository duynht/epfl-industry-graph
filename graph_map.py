import pandas as pd
import argparse
import pickle
import json
import os
from glob import glob
import csv

def parse_graph(datapath):
    node_counter = -1
    node_dict = {}
    node_type_dict = {}
    edge_list = []
    # extracted_path = os.path.join(datapath, 'extracted/**')
    edge_df = pd.read_csv(os.path.join(datapath, 'relationships.csv'))
    company_df = pd.read_csv(os.path.join(datapath, 'companies.csv'))
    field_df = pd.read_csv(os.path.join(datapath, 'fields.csv'))
    node_dict = {name:id for id, name in enumerate(set(edge_df['company']).union(set(edge_df['field'])))}
    edge_list = list(zip(edge_df['company'], edge_df['field']))
    edge_list = [(node_dict[edge[0]], node_dict[edge[1]]) for edge in edge_list]

    for company in edge_df['company']:
        node_type_dict[node_dict[company]] = 'company'
    for field in edge_df['field']:
        node_type_dict[node_dict[field]] = 'field'

    output_dir = os.path.join(datapath,'parsed-graph')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, 'pt_graph.csv')
    with open(filepath,'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['id_1','id_2'])
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
    parser.add_argument('-d', '--datapath', 
                        default='../data/csv',
                        type=str,
                        help='path to data directory (default is "../data/csv")')

    args = parser.parse_args()

    parse_graph(args.datapath)
    print('Done parsing!')