import json
import pickle
import json
import os
from glob import glob
from gensim.models import Word2Vec as wv
import numpy as np
import pandas as pd
from collections import defaultdict
from enum import Enum
import re
import faiss

class NodeType(Enum):
    company = 0
    field = 1

class QueryType(Enum):
    company2company = 0,
    company2field = 1,
    field2company = 2,
    field2field = 3

class Evaluator:
    persona_map = defaultdict(set)
    inv_persona_map = {}
    db_index = {}
    evaluate_set = defaultdict(lambda: defaultdict(set))
    evaluate_list = defaultdict(lambda: defaultdict(list))
    index_map = defaultdict(list)

    def __init__(self, top_k, use_gpu=False):
        self.top_k = top_k

        with open('../data/parsed-graph/node_type_dict.pkl','rb') as f:
            self.node_type_dict = pickle.load(f)

        with open('../data/parsed-graph/node_dict.pkl','rb') as f:
            self.node_dict = pickle.load(f)

        with open('../data/parsed-graph/str2id_dict.pkl','rb') as f:
            self.inv_node_dict = pickle.load(f)

        with open('../data/embeddings/persona_map.txt','r') as f:
            for line in f:
                persona_node, original_node = map(int, line.split())

                self.persona_map[original_node].add(persona_node)
                self.inv_persona_map[persona_node] = original_node

                if original_node not in self.inv_persona_map:
                    self.inv_persona_map[original_node] = original_node
                
        self.persona_emb = wv.load_word2vec_format('../data/embeddings/persona.embedding')
   
        for node_type in NodeType:    
            self.db_index[node_type] =  faiss.IndexFlatL2(self.persona_emb.vector_size)
            if (use_gpu):
                self.db_index[node_type] = faiss.index_cpu_to_gpu(faiss.StandardGpuResources() , 0, self.db_index[node_type])
            
            embs = []
            for key in sorted(list(self.persona_emb.vocab)):
                if self.node_type_dict[self.inv_persona_map[int(key)]] == node_type.name:
                    self.index_map[node_type].append(int(key))
                    embs.append(self.persona_emb[key])
            
            db = np.array(embs)
            self.db_index[node_type].add(db)

        self.load_truth(filepath='../data/truth/company_related_company.json', src_type='company', dst_type='company')
        self.load_truth(filepath='../data/truth/company_related_technology.json', src_type='company', dst_type='field')
        self.load_truth(filepath='../data/truth/technology_company.json', src_type='field', dst_type='company')
        self.load_truth(filepath='../data/truth/technology_resinst.json', src_type='field', dst_type='company')    

        for query_type in QueryType:
            for key in self.evaluate_list[query_type]:
                self.evaluate_list[query_type][key].sort()

    def load_truth(self, filepath, src_type, dst_type):
        query_type = QueryType[src_type+'2'+dst_type]
        with open(filepath,encoding='utf8') as f:
            for line in f: 
                try:
                    data = json.loads(line)
                    for key, entries in data.items():
                        if isinstance(entries, list):
                            key = ' '.join(re.sub(r'[^a-zA-Z\d,]',' ', key.lower()).split())

                            self.evaluate_set[query_type][key] = self.evaluate_set[query_type][key].union(
                                {' '.join(re.sub(r'[^a-zA-Z\d,]',' ', entry['value']['name'].lower()).split()) for entry in entries}
                            )

                            self.evaluate_list[query_type][key] += [(entry['score'],' '.join(re.sub(r'[^a-zA-Z\d,]',' ', entry['value']['name'].lower()).split())) for entry in entries]
    
                except ValueError as e:
                    pass 

    def evaluate_node(self, node_str, src_type, dst_type):        
        node_str = ' '.join(re.sub(r'[^a-zA-Z\d,]',' ', node_str.lower()).split())
        query_type = QueryType[src_type.name+'2'+dst_type.name]

        if query_type.name == 'field2field':
            node_id = self.inv_node_dict[node_str]
            
            num_persona = len(self.persona_map[node_id])

            queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
            distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

            distances = [dist for batch in distances for dist in batch]
            indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

            neighbors = pd.DataFrame(list(zip(indices, distances)))
            neighbors = neighbors.groupby(0).min().to_dict()[1].items()
            neighbors = sorted(neighbors, key=lambda x: x[1])                

            neighbors = [self.node_dict[neighbor_id] for neighbor_id, _ in neighbors[:self.top_k]]     

            return 0 , 0, neighbors, []

        else:
            if node_str not in self.inv_node_dict:
                if not self.evaluate_set[query_type][node_str]:
                    return 0, 0, [], []
                else:
                    return 0, 0, [], [name for _, name in self.evaluate_list[query_type][node_str][:self.top_k]]

            node_id = self.inv_node_dict[node_str]
            
            num_persona = len(self.persona_map[node_id])

            queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
            distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

            distances = [dist for batch in distances for dist in batch]
            indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

            neighbors = pd.DataFrame(list(zip(indices, distances)))
            neighbors = neighbors.groupby(0).min().to_dict()[1].items()
            neighbors = sorted(neighbors, key=lambda x: x[1])                

            neighbors = [self.node_dict[neighbor_id] for neighbor_id, _ in neighbors[:self.top_k]]     

            precision = len([neighbor for neighbor in neighbors if neighbor in self.evaluate_set[query_type][node_str] ]) / len(neighbors)

            if not self.evaluate_set[query_type][node_str]:
                recall = 0
            else:
                recall = len([neighbor for neighbor in neighbors if neighbor in self.evaluate_set[query_type][node_str] ]) / len(self.evaluate_set[query_type][node_str])

            return precision, recall, neighbors, [name for _, name in self.evaluate_list[query_type][node_str][:self.top_k]]


if __name__ == "__main__":
    eval = Evaluator(top_k = 10)

    evaluation_result = {}

    with open('../results/mAP@10.txt', 'w') as f:
        for src_type in NodeType:
            for dst_type in NodeType:
                if (src_type.name == 'company'):
                    continue

                if (src_type.name == 'field' and dst_type.name == 'field'):
                    continue

                query_type = QueryType[src_type.name+'2'+dst_type.name]
                mAP = 0
                query_result = {}

                for node_str in eval.evaluate_set[query_type]:
                    precision, recall, neighbors, ground_truth = eval.evaluate_node(node_str, src_type, dst_type)
                    query_result[node_str] = {'precision' : precision, 'recall' : recall, 'neighbors' : neighbors, 'ground truth' : ground_truth}
                    mAP += precision
                
                evaluation_result[query_type.name] = query_result
                
                mAP /= len(eval.evaluate_set[query_type])
                
                f.write(''.join([query_type.name,' : mAP = ', str(mAP)]))
    
    json.dump(evaluation_result, open('../results/evaluation_result.json', 'w', encoding='utf-8'), indent=2)