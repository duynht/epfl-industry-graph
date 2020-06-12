import pickle
import json
import os
from glob import glob
from gensim.models import KeyedVectors as wv
import numpy as np
from collections import defaultdict
import faiss

fields_companies = defaultdict(list)
fields_fields = defaultdict(list)
companies_companies = defaultdict(list)
persona_map = defaultdict(list)
persona_inv_map = {}

threshold = 0.6

res = faiss.StandardGpuResources()

with open('data/parsed-graph/node_type_dict.pkl','rb') as f:
    node_type_dict = pickle.load(f)

with open('data/parsed-graph/node_dict.pkl','rb') as f:
    node_dict = pickle.load(f)

with open('data/embeddings/persona_map.txt','r') as f:
    for line in f:
        persona_node, original_node = map(int, line.split())
        persona_map[original_node].append(persona_node)
        persona_inv_map[persona_node] = original_node

persona_emb = wv.load_word2vec_format('data/embeddings/persona.embedding')

# quantizer = faiss.IndexFlatL2(persona_emb.vectors.shape[1])
# index_ivf = faiss.IndexIVFFlat(quantizer, persona_emb.vectors.shape[1])
companies_index_l2 =  faiss.IndexFlatL2(persona_emb.vectors.shape[1])
companies_index_l2 = faiss.index_cpu_to_gpu(res, 0, companies_index_l2)
companies_index_l2.add([persona_emb[key] for _, key in enumerate(persona_emb.vocab) if node_type_dict[int(key)] == 'company'])

fields_index_l2 =  faiss.IndexFlatL2(persona_emb.vectors.shape[1])
fields_index_l2 = faiss.index_cpu_to_gpu(res, 0, companies_index_l2)
fields_index_l2.add([persona_emb[key] for _, key in enumerate(persona_emb.vocab) if node_type_dict[int(key)] == 'field'])


count = 0

for node_id, node_str in node_dict.items():
    if count == 10:
        break
    if (node_type_dict[node_id] == 'company'): 
        count += 1
        print(count,' companies')

    # TODO: Sorted results 

    neighbors = []
    for persona in persona_map[node_id]:
        top_5 = companies_index_l2.search(persona_emb[str(node_id)], 5)
        

    for neighbor_id, neighbor_str in node_dict.items():
        # if node_id == neighbor_id: 
        #     continue
        # if (node_type_dict[node_id] == 'company' and node_type_dict[neighbor_id] == 'field'):
        #     continue
        
        # score = []
        # for persona in persona_map[node_id]:
        #     for neighbor_persona in persona_map[neighbor_id]:
        #         score.append(persona_emb.similarity(str(persona), str(neighbor_persona)))

        
        # score = max(score)
        
        # TODO: More efficient scoring

        if (score > threshold):
            if (node_type_dict[node_id] == 'company' and node_type_dict[neighbor_id] == 'company'):
                companies_companies[node_str].append(neighbor_str)
        
            elif (node_type_dict[node_id] == 'field' and node_type_dict[neighbor_id] == 'company'):
                fields_companies[node_str].append(neighbor_str)
            
            elif (node_type_dict[node_id] == 'field' and node_type_dict[neighbor_id] == 'field'):
                fields_fields[node_str].append(neighbor_str)

with open('results/10-companies-companies.json','w',encoding='utf-8') as f:
    json.dump(companies_companies, f, indent=2)

with open('results/10-fields-companies.json','w',encoding='utf-8') as f:
    json.dump(fields_companies, f, indent=2)

with open('results/10-fields-fields.json','w',encoding='utf-8') as f:
    json.dump(fields_fields, f, indent=2)
