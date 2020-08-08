import pickle
import json
import os
from glob import glob
from gensim.models import Word2Vec as wv
import numpy as np
from collections import defaultdict

class NaiveDB:
    def __init__(self):
        with open('../data/parsed-graph/node_type_dict.pkl','rb') as f:
            self.node_type_dict = pickle.load(f)

        with open('../data/parsed-graph/str2id_dict.pkl','rb') as f:
            self.node_dict = pickle.load(f)

        self.persona_map = defaultdict(list)
        with open('../data/embeddings/persona_map.txt','r') as f:
            for line in f:
                persona_node, original_node = map(int, line.split())
                self.persona_map[original_node].append(persona_node)
        
        self.persona_emb = wv.load_word2vec_format('../data/embeddings/persona.embedding')

        self.field_types = {'company', 'field'}

        self.threshold = 0.6

    
    def get_related_nodes(self, node_str, src_type, dest_type):
        try:
            assert src_type in self.field_types
        except AssertionError as e:
            e.args += (src_type)
            raise

        try:
            assert dest_type in self.field_types
        except AssertionError as e:
            e.args += (dest_type)
            raise
        try:
            assert self.node_type_dict[self.node_dict[node_str]] == src_type
        except AssertionError as e:
            e.args += (node_str, self.node_dict[node_str], src_type)
            raise

        node_id = self.node_dict[node_str]

        related_list = []

        for neighbor_str, neighbor_id in self.node_dict.items():
            if not (self.node_type_dict[neighbor_id] == dest_type):
                continue
            score = [self.persona_emb.similarity(str(persona), str(neighbor_persona))
                        for persona in self.persona_map[node_id] 
                            for neighbor_persona in self.persona_map[neighbor_id]]
            score = max(score)
            if (score > self.threshold):
                related_list.append(neighbor_str)
        
        return related_list
    