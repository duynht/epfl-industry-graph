import os.path as osp
import pandas as pd 
import torch 
from twitter_swiss_actors import TwitterSwissActors as TSA
from gensim.models import KeyedVectors
from gensim.similarities.annoy import AnnoyIndexer
# import networkx as nx
import json
import ast

def precision_at_k(result_list, truth_set, k=40):
    if len(result_list) > k: 
        result_list = result_list[:k]
    
    match_count = 0
    for result in result_list:
        if result in truth_set:
            match_count += 1
    
    return match_count / k

def recall_at_k(result_list, truth_set, k=40):
    if len(result_list) > k: 
        result_list = result_list[:k]
    
    match_count = 0
    for result in result_list:
        if result[0] in truth_set:
            match_count += 1
    
    return match_count / len(truth_set) if len(truth_set) > 0 else match_count

def set_discard(s, val):

    set(s).discard(val)
    return s

def company2company_eval(result, truth_filename, walk_length):
    crunch_df = pd.read_csv(truth_filename, usecols=['Twitter', 'Related'], dtype={'Twitter': 'string'})
    crunch_df = crunch_df.rename(columns = {'Twitter': 'username', 'Related': 'crunchbase'})
    crunch_df = crunch_df[crunch_df['crunchbase'].notnull()]
    crunch_df = crunch_df[crunch_df['username'].notnull()]
    crunch_df['crunchbase'] = crunch_df['crunchbase'].apply(lambda x: '{}' if x == 'set()' else x)
    crunch_df['crunchbase'] = crunch_df['crunchbase'].map(ast.literal_eval)
    crunch_df['crunchbase'] = crunch_df['crunchbase'].apply(lambda x: set_discard(x, ''))

    result_df = pd.DataFrame({'username': result.keys(), 'retrieved': result.values()})
    result_df = result_df.merge(crunch_df, on='username')
    result_df['precision@40'] = result_df.apply(lambda row: precision_at_k(row.retrieved, row.crunchbase), axis=1)
    result_df['recall@40'] = result_df.apply(lambda row: recall_at_k(row.retrieved, row.crunchbase), axis=1)

    result_df.to_csv(osp.join('results', '_'.join(['company2company', 'metapath2vec', str(walk_length), truth_filename.split('/')[-1],'result.csv'])), index=False)
    result_df.to_csv(osp.join('~/onedrive/EPFL/results', '_'.join(['company2company', 'metapath2vec', str(walk_length), truth_filename.split('/')[-1], 'result.csv'])), index=False)
    result_df[['username', 'precision@40', 'recall@40']].to_csv(osp.join('results', '_'.join(['company2company', 'metapath2vec', str(walk_length), truth_filename.split('/')[-1], 'short', 'result.csv'])), index=False)
    result_df[['username', 'precision@40', 'recall@40']].to_csv(osp.join('~/onedrive/EPFL/results', '_'.join(['company2company', 'metapath2vec', str(walk_length), truth_filename.split('/')[-1], 'short', 'result.csv'])), index=False)



if __name__ == "__main__":

    path = '../data/twitter_swiss_actors/processed'
    k = 40
    type_set = {'company', 'field'}
    emb_dict = {}

    for walk_length in [4, 8, 16, 32]:
        for node_type in type_set:
            emb_dict[node_type] = KeyedVectors.load(osp.join(path, f'{node_type}_metapath2vec_{walk_length}'))

        # indexer = {}
        # for type, emb in emb_dict.items():
        #     indexer[type] = AnnoyIndexer(emb)

        query_set = {'company2company', 'company2field', 'field2company'}
        # query_set = {'company2company'}

        graphs = {}
        for query in query_set:
            graphs[query] = {}
            src_type = query.split('2')[0]
            dst_type = query.split('2')[-1]
            emb = emb_dict[src_type]
            for u in emb.index_to_key:
                # top_k = emb_dict[dst_type].most_similar([emb.get_vector(u)], topn=k, indexer=indexer[dst_type])
                top_k = emb_dict[dst_type].similar_by_vector(emb.get_vector(u), topn=k)
                graphs[query][u] = [v for v,_ in top_k]

            json.dump(graphs[query], open(osp.join('results', f'metapath2vec_{walk_length}_{query}.json'), "w")) #### output networkx graph

        # result = {}
        #company2company
        result = graphs['company2company']
        company2company_eval(result, '~/data/truth/crunchbase/crunchbase.csv', walk_length)
        company2company_eval(result, '~/data/truth/crunchbase/swiss_crunchbase.csv', walk_length)
        


    
    





