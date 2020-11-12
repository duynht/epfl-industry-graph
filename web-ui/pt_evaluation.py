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
import requests
import argparse
from matplotlib import pyplot as plt


def concept_lookup(id):
    concept = 'unknown'
    url = 'http://localhost:8090/service/kb/concept/'+id
    with requests.get(url) as resp:
        try:
            resp.raise_for_status()
            try:
                anno = resp.json()
                try:
                    concept = anno['preferredName']
                except KeyError as e:
                    print(e.__class__, e, anno)
            except json.decoder.JSONDecodeError as e:
                print(e.__class__, e, resp.text)
        except requests.exceptions.HTTPError as e:
            print(e.__class__, e, e.response, e.re.request, text)
    
    return concept.lower()

def term_lookup(term):
    url = 'http://localhost:8090/service/kb/term/'+term 
    url = url.encode()

    wikidataId = 'NULL'
    pageid = None

    with requests.get(url) as resp:
        try:
            resp.raise_for_status()
            try:
                anno = resp.json()
                try:
                    pageid = anno['senses'][0]['pageid']
                except IndexError as e:
                    print(e.__class__, e, anno)
            except json.decoder.JSONDecodeError as e:
                print(e.__class__, e, resp.text)             
        except requests.exceptions.HTTPError as e:
            print(e.__class__, e, e.response, e.re.request, text)

    if pageid:
        url = 'http://localhost:8090/service/kb/concept/'+str(pageid)
        with requests.get(url) as resp:
            try:
                resp.raise_for_status()
                anno = resp.json()
                try:
                    wikidataId = anno['wikidataId']
                except KeyError as e:
                    print(e.__class__, e, anno)
            except requests.exceptions.HTTPError as e:
                print(e, e.response, e.re.request, text)
    
    return wikidataId

# def get_wikidataId(term):
#     page = wptools.page(term).get_parse()
#     wikidataId = page.data['wikibase']
#     return wikidataId

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
    persona_emb = {}

    def __init__(self, datapath, top_k, use_gpu=False):
        graph_dir = os.path.join(datapath, 'csv/parsed-graph')
        emb_dir = os.path.join(datapath, 'embeddings')
        truth_dir = os.path.join(datapath, 'truth')

        self.top_k = top_k

        with open(os.path.join(graph_dir,'pt_node_type_dict.pkl'),'rb') as f:
            self.node_type_dict = pickle.load(f)

        with open(os.path.join(graph_dir, 'pt_node_dict.pkl'),'rb') as f:
            self.node_dict = pickle.load(f)

        with open(os.path.join(graph_dir, 'pt_str2id_dict.pkl'),'rb') as f:
            self.inv_node_dict = pickle.load(f)
        
        field_df = pd.read_csv(os.path.join(datapath,'csv/fields.csv'))
        self.field_dict = field_df.set_index('wikidataId')['label'].to_dict()

        # with open('~/data/embeddings/persona_map.txt','r') as f:
        #     for line in f:
        #         persona_node, original_node = map(int, line.split())

        #         self.persona_map[original_node].add(persona_node)
        #         self.inv_persona_map[persona_node] = original_node

        #         if original_node not in self.inv_persona_map:
        #             self.inv_persona_map[original_node] = original_node
                
        # self.persona_emb = wv.load_word2vec_format('~/data/embeddings/persona.embedding')
   
        # for node_type in NodeType:    
        #     self.db_index[node_type] =  faiss.IndexFlatL2(self.persona_emb.vector_size)
        #     if (use_gpu):
        #         self.db_index[node_type] = faiss.index_cpu_to_gpu(faiss.StandardGpuResources() , 0, self.db_index[node_type])
            
        #     embs = []
        #     for key in sorted(list(self.persona_emb.vocab)):
        #         if self.node_type_dict[self.inv_persona_map[int(key)]] == node_type.name:
        #             self.index_map[node_type].append(int(key))
        #             embs.append(self.persona_emb[key])
            
        #     db = np.array(embs)
        #     self.db_index[node_type].add(db)
        with open(os.path.join(emb_dir, 'pt_persona_map.json'), 'r') as f:
            self.inv_persona_map = json.load(f)
        self.inv_persona_map = {int(key):int(value) for key, value in self.inv_persona_map.items()}
        for persona_node, original_node in self.inv_persona_map.items():
            self.persona_map[original_node].add(persona_node)
            # if original_node not in self.persona_map[original_node]:
            #     self.persona_map[original_node].add(original_node)

        self.persona_emb_df = pd.read_csv(os.path.join(emb_dir, 'persona_embedding.csv'))
        self.persona_emb_df.set_index('id')
        
        for node_type in NodeType:  
            self.db_index[node_type] =  faiss.IndexFlatL2(len(self.persona_emb_df.columns) -1)
            if (use_gpu):
                self.db_index[node_type] = faiss.index_cpu_to_gpu(faiss.StandardGpuResources() , 0, self.db_index[node_type])
            
            embs = []
            for key in self.persona_emb_df.index:
                if self.node_type_dict[self.inv_persona_map[int(key)]] == node_type.name:
                    self.index_map[node_type].append(int(key))
                    self.persona_emb[str(key)] = self.persona_emb_df.loc[key, self.persona_emb_df.columns != 'id'].to_numpy(dtype=np.float32)
                    embs.append(self.persona_emb[str(key)])
            
            db = np.array(embs)
            self.db_index[node_type].add(db)

        self.load_truth(os.path.join(truth_dir, 'company_related_company.json'), src_type='company', dst_type='company')
        self.load_truth(os.path.join(truth_dir, 'company_related_technology.json'), src_type='company', dst_type='field')
        self.load_truth(os.path.join(truth_dir, 'technology_company.json'), src_type='field', dst_type='company')
        self.load_truth(os.path.join(truth_dir, 'technology_resinst.json'), src_type='field', dst_type='company')    

        for query_type in QueryType:
            for key in self.evaluate_list[query_type]:
                self.evaluate_list[query_type][key].sort()

    # def load_truth(self, filepath, src_type, dst_type):
    #     query_type = QueryType[src_type+'2'+dst_type]
    #     with open(filepath,encoding='utf8') as f:
    #         for line in f: 
    #             try:
    #                 data = json.loads(line)
    #                 for key, entries in data.items():
    #                     if isinstance(entries, list):
    #                         key = key.replace('_', ' ')

    #                         self.evaluate_set[query_type][key] = self.evaluate_set[query_type][key].union(
    #                             {entry['value']['name'].lower().replace('_', ' ') for entry in entries}
    #                         )

    #                         self.evaluate_list[query_type][key] += [(entry['score'], entry['value']['name'].lower().replace('_', ' ')) for entry in entries]
    
    #             except ValueError as e:
    #                 pass 

    def load_truth(self, filepath, src_type, dst_type):
        query_type = QueryType[src_type+'2'+dst_type]
        with open(filepath,encoding='utf8') as f:
            for line in f: 
                try:
                    data = json.loads(line)
                    for key, entries in data.items():
                        if src_type == 'field':
                            key = key.replace('_', ' ').lower()
                        
                        if dst_type == 'field':
                            for entry in entries:
                                entry['value']['uid'] = entry['value']['uid'].replace('_', ' ').lower()
                                entry['value']['name'] = entry['value']['name'].replace('_', ' ').lower()

                        if isinstance(entries, list):
                            self.evaluate_set[query_type][key] = self.evaluate_set[query_type][key].union(
                                {entry['value']['uid'] for entry in entries}
                            )

                            self.evaluate_list[query_type][key] += [(entry['score'], (entry['value']['uid'], entry['value']['name'])) for entry in entries]

                except ValueError as e:
                    pass 

    def company_to_company(self, zefix_uid):
        if not zefix_uid in self.inv_node_dict:
            return []

        dst_type = NodeType.company
        node_id = self.inv_node_dict[zefix_uid]
        num_persona = len(self.persona_map[node_id])

        queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
        distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

        distances = [dist for batch in distances for dist in batch]
        indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

        neighbors = pd.DataFrame(list(zip(indices, distances)))
        neighbors = neighbors.groupby(0).min().to_dict()[1].items()
        neighbors = sorted(neighbors, key=lambda x: x[1])    
        neighbors = [self.node_dict[neighbor_id] for neighbor_id, _ in neighbors[:self.top_k]] 

        return neighbors           

    def company_to_field(self, zefix_uid):
        if not zefix_uid in self.inv_node_dict:
            return []
        dst_type = NodeType.field
        node_id = self.inv_node_dict[zefix_uid]
        num_persona = len(self.persona_map[node_id])

        queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
        distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

        distances = [dist for batch in distances for dist in batch]
        indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

        neighbors = pd.DataFrame(list(zip(indices, distances)))
        neighbors = neighbors.groupby(0).min().to_dict()[1].items()
        neighbors = sorted(neighbors, key=lambda x: x[1])    
        neighbors = [self.field_dict[self.node_dict[neighbor_id]] for neighbor_id, _ in neighbors[:self.top_k]] 

        return neighbors 

    def field_to_company(self, field_name):
        if not field_name in self.inv_node_dict:
            return []
        dst_type = NodeType.company
        node_id = self.inv_node_dict[field_name]
        num_persona = len(self.persona_map[node_id])

        queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
        distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

        distances = [dist for batch in distances for dist in batch]
        indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

        neighbors = pd.DataFrame(list(zip(indices, distances)))
        neighbors = neighbors.groupby(0).min().to_dict()[1].items()
        neighbors = sorted(neighbors, key=lambda x: x[1])    
        neighbors = [self.node_dict[neighbor_id] for neighbor_id, _ in neighbors[:self.top_k]] 

        return neighbors

    def evaluate_node(self, query_type, node_id):
        if query_type == QueryType.company2company:            
            # company_list = self.company_to_company(node_id)
            # neighbor_list = [company['dst.uid'] for company in company_list]
            neighbor_list = self.company_to_company(node_id)
                
        elif query_type == QueryType.company2field:
            # field_list = self.company_to_field(node_id)
            # neighbor_list = [field['field.label'] for field in field_list]
            neighbor_list = self.company_to_field(node_id)

        # elif query_type == QueryType.field2company:
        else:
            # company_list = self.field_to_company(node_id)
            # neighbor_list = [company['dst.uid'] for company in company_list]
            neighbor_list = self.field_to_company(node_id)
        
        precision = len([neighbor for neighbor in neighbor_list if neighbor in self.evaluate_set[query_type][node_id]]) / self.top_k
        try:
            recall = len([neighbor for neighbor in neighbor_list if neighbor in self.evaluate_set[query_type][node_id]]) / len(self.evaluate_set[query_type][node_id])
        except ZeroDivisionError:
            recall = 0.0

        return precision, recall, neighbor_list, [name for _, name in self.evaluate_list[query_type][node_id][:self.top_k]]
        


    # def evaluate_node(self, ori_node_str, src_type, dst_type, zefix_uid = None):        
    #     # node_str = ' '.join(re.sub(r'[^a-zA-Z\d,]',' ', node_str.lower()).split())
    #     ori_node_str = ori_node_str.replace('_', ' ')
    #     node_str = ori_node_str

    #     if src_type.name == 'field':
    #         node_str = term_lookup(ori_node_str)

    #     query_type = QueryType[src_type.name+'2'+dst_type.name]

    #     node_str = node_str.lower()

    #     if query_type.name == 'field2field':
    #         node_id = self.inv_node_dict[node_str]
            
    #         num_persona = len(self.persona_map[node_id])

    #         queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
    #         distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

    #         distances = [dist for batch in distances for dist in batch]
    #         indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

    #         neighbors = pd.DataFrame(list(zip(indices, distances)))
    #         neighbors = neighbors.groupby(0).min().to_dict()[1].items()
    #         neighbors = sorted(neighbors, key=lambda x: x[1])                

    #         neighbors = [concept_lookup(self.node_dict[neighbor_id]) for neighbor_id, _ in neighbors[:self.top_k]]     

    #         return 0 , 0, neighbors, []
    #     else:
    #         if node_str not in self.inv_node_dict:
    #             if zefix_uid is not None:
    #                 zefix_uid = re.sub(r'[^a-zA-Z\d]','', zefix_uid.upper())
    #                 ori_node_str = zefix_uid
                
    #             # import pdb; pdb.set_trace()
    #             if not self.evaluate_set[query_type][ori_node_str]:
    #                 return 0, 0, [], []
    #             else:
    #                 return 0, 0, [], [name for _, name in self.evaluate_list[query_type][ori_node_str][:self.top_k]]

    #         node_id = self.inv_node_dict[node_str]

    #         if zefix_uid is not None:
    #             zefix_uid = re.sub(r'[^a-zA-Z\d]','', zefix_uid.upper())
            
    #         num_persona = len(self.persona_map[node_id])

    #         queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
    #         distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

    #         distances = [dist for batch in distances for dist in batch]
    #         indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

    #         neighbors = pd.DataFrame(list(zip(indices, distances)))
    #         neighbors = neighbors.groupby(0).min().to_dict()[1].items()
    #         neighbors = sorted(neighbors, key=lambda x: x[1])                

    #         if (dst_type.name == 'field'):
    #             neighbors = [concept_lookup(self.node_dict[neighbor_id]) for neighbor_id, _ in neighbors[:self.top_k]]     
    #         else:
    #             neighbors = [self.node_dict[neighbor_id] for neighbor_id, _ in neighbors[:self.top_k]]

    #         precision = len([neighbor for neighbor in neighbors if neighbor in self.evaluate_set[query_type][ori_node_str] ]) / len(neighbors)

    #         if not self.evaluate_set[query_type][ori_node_str]:
    #             recall = 0
    #         else:
    #             recall = len([neighbor for neighbor in neighbors if neighbor in self.evaluate_set[query_type][ori_node_str] ]) / len(self.evaluate_set[query_type][ori_node_str])
    #         # import pdb; pdb.set_trace()
    #         return precision, recall, neighbors, [name for _, name in self.evaluate_list[query_type][ori_node_str][:self.top_k]]


def plot_pr(pr_dict, top_k):
    # textstr = 'mAP@10={mAP:.2f} \nmatched_mAP@10={mOM:.2f}'.format(mAP=mAP, mOM=mAP_over_matched)
    color_list = ['red', 'green', 'blue']
    line_list = []
    label_list = []
    for query_type, color in zip(pr_dict.keys(), color_list):
        precision_list = pr_dict[query_type]['precision']
        recall_list = pr_dict[query_type]['recall']
        precision_list = [precision for _,precision in sorted(zip(recall_list, precision_list))]
        recall_list = sorted(recall_list)
        l, = plt.step(recall_list, precision_list, color=color, alpha=0.7)
        line_list.append(l)
        # label_list.append('{query_type} (mAP = {mAP:.2f} mAPofMatched = {mAPofMatched:.2f}'
        #                     .format(query_type = query_type,
        #                             mAP=pr_dict[query_type]['mAP'],
        #                             mAPofMatched=pr_dict[query_type]['mAPofMatched']))
        label_list.append(query_type)

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlabel('Recall@{top_k}'.format(top_k=top_k))
    plt.ylabel('Precision@{top_k}'.format(top_k=top_k))
    plt.ylim([0.0, 0.65])
    plt.xlim([0.0, 0.65])
    # fig.title('Precision-recall @ {top_k}')
    # plt.text(0.05, 0.95, textstr)
    plt.legend(line_list, label_list, loc=(0, -.42), prop=dict(size=10))
    plt.savefig('prcurve_{top_k}.pdf'.format(top_k=top_k))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-d', '--datapath', 
#                         default='../../data',
#                         type=str,
#                         help='path to data directory (default is "../../data")')
    
#     parser.add_argument('-rp', '--result_path', 
#                         default='../results',
#                         type=str,
#                         help='path to result directory (default is "../results")')
    
#     parser.add_argument('-k', '--top_k', 
#                         default=10,
#                         type=int,
#                         help='top k nearest neighbors (default is 10)')

#     parser.add_argument('-gpu', '--use_gpu',
#                         default=False,
#                         action='store_true',
#                         help='use GPU for indexing (defaulted to not using)')

#     args = parser.parse_args()


#     if not os.path.exists(args.result_path):
#         os.makedirs(args.result_path)


#     eval = Evaluator(datapath=args.datapath, top_k=args.top_k, use_gpu=args.use_gpu)

#     evaluation_result = {}

#     with open(os.path.join(args.result_path, 'mAP@10.txt'), 'w') as f:
#         for src_type in NodeType:
#             for dst_type in NodeType:
#                 if (src_type.name == 'company'):
#                     continue

#                 if (src_type.name == 'field' and dst_type.name == 'field'):
#                     continue

#                 query_type = QueryType[src_type.name+'2'+dst_type.name]
#                 mAP = 0
#                 query_result = {}
#                 # import pdb; pdb.set_trace()
#                 for node_str in list(eval.evaluate_set[query_type].keys()):
#                     precision, recall, neighbors, ground_truth = eval.evaluate_node(node_str, src_type, dst_type)
#                     query_result[node_str] = {'precision' : precision, 'recall' : recall, 'neighbors' : neighbors, 'ground truth' : ground_truth}
#                     mAP += precision
                
#                 evaluation_result[query_type.name] = query_result
                
#                 mAP /= len(eval.evaluate_set[query_type])
                
#                 f.write(''.join([query_type.name,' : mAP = ', str(mAP)]))
    
#     json.dump(evaluation_result, open(os.path.join(args.result_path, 'evaluation_result.json'), 'w', encoding='utf-8'), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', 
                        default='../../data',
                        type=str,
                        help='path to data directory (default is "../../data")')
    
    parser.add_argument('-rp', '--result_path', 
                        default='../results',
                        type=str,
                        help='path to result directory (default is "../results")')
    
    parser.add_argument('-k', '--top_k', 
                        default=10,
                        type=int,
                        help='top k nearest neighbors (default is 10)')

    parser.add_argument('-gpu', '--use_gpu',
                        default=False,
                        action='store_true',
                        help='use GPU for indexing (defaulted to not using)')

    args = parser.parse_args()


    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    eval = Evaluator(datapath=args.datapath, top_k=args.top_k, use_gpu=args.use_gpu)

    evaluation_result = {}

    pr_dict = {}

    for src_type in NodeType:
        for dst_type in NodeType:
            query_type = QueryType[src_type.name+'2'+dst_type.name]

            if query_type == QueryType.field2field:
                continue

            cum_prec = 0
            matched_count = 0
            query_result = {}
            pr_dict[query_type.name] = {}
            pr_dict[query_type.name]['precision'] = []
            pr_dict[query_type.name]['recall'] = []
            for node_id in list(eval.evaluate_set[query_type].keys()):
                precision, recall, neighbors, ground_truth = eval.evaluate_node(query_type, node_id)
                pr_dict[query_type.name]['precision'].append(precision)
                pr_dict[query_type.name]['recall'].append(recall)

                if precision > 0.0 or recall > 0.0:
                    query_result[node_id] = {'precision' : precision, 'recall' : recall, 'neighbors' : neighbors, 'ground truth' : ground_truth}
                    matched_count += 1
                    cum_prec += precision
            
            evaluation_result[query_type.name] = query_result
            
            pr_dict[query_type.name]['mAP'] = cum_prec/len(eval.evaluate_set[query_type])
            pr_dict[query_type.name]['mAPofMatched'] = cum_prec/ (matched_count if matched_count > 0 else 1)
            # plot_pr(precision_list, recall_list, mAP_dict[query_type.name]['mAP'], mAP_dict[query_type.name]['mAP_over_matched'], query_type, args.top_k)
            plot_pr(pr_dict, args.top_k)
    json.dump(evaluation_result, open(os.path.join(args.result_path, 'neo4j_evaluation_result.json'), 'w', encoding='utf-8'), indent=2)
    json.dump(pr_dict, open(os.path.join(args.result_path, 'neo4j_mAP@{k}.txt'.format(k=args.top_k)), 'w', encoding='utf-8'), indent=2)

