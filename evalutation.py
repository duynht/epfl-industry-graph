import ast
import pickle
import json
import os
from glob import glob
from gensim.models import KeyedVectors as wv
import numpy as np
import pandas as pd
from collections import defaultdict
import faiss

class Evaluator:
    persona_map = defaultdict(list)
    inv_persona_map = {}
    db_index = {}

    def __init__(self, top_k):
        self.top_k = top_k

        with open('data/parsed-graph/node_type_dict.pkl','rb') as f:
            self.node_type_dict = pickle.load(f)

        with open('data/parsed-graph/node_dict.pkl','rb') as f:
            self.node_dict = pickle.load(f)

        with open('data/parsed-graph/str2id_dict.pkl','rb') as f:
            self.inv_node_dict = pickle.load(f)

        with open('data/embeddings/persona_map.txt','r') as f:
            for line in f:
                persona_node, original_node = map(int, line.split())
                self.persona_map[original_node].append(persona_node)
                self.inv_persona_map[persona_node] = original_node
                if not (original_node in inv_persona_map):
                    self.inv_persona_map[original_node] = original_node

        self.persona_emb = wv.load_word2vec_format('data/embeddings/persona.embedding')

        res = faiss.StandardGpuResources()    
        for (node_type in ('company', 'field')):    
            self.db_index[node_type] =  faiss.IndexFlatL2(self.persona_emb.vector_size)
            self.db_index[node_type] = faiss.index_cpu_to_gpu(res, 0, self.db_index[node_type])
            db = np.array([persona_emb[str(key)] for key in sorted(list(persona_emb.vocab)) if node_type_dict[inv_persona_map[int(key)]] == node_type])
            self.db_index[node_type].add(db)

    def evaluate(self,filepath, src_type, dst_type):
        result_dict = defaultdict(list)

        result_dict[node_str] = [node_dict[neighbor_id] for neighbor_id, _ in neighbors[:top_k]]

        evaluate_set = {}

        for node_str in evaluate_set:
            node_id = inv_node_dict[node_str]
            
            num_persona = len(persona_map[node_id])
            neighbors = {}
            queries = np.array([persona_emb[str(persona)] for persona in persona_map[node_id]])
            distances, indices = fields_index_l2.search(queries, num_persona + top_k)

            distances = [dist for batch in distances for dist in batch]
            indices = [inv_persona_map[id] for index in indices for id in index]

            neighbors = pd.DataFrame(list(zip(indices, distances)))
            neighbors = neighbors.groupby(0).min().to_dict()[1].items()
            neighbors = sorted(neighbors, key=lambda x: x[1])                       

    

    