import os
from glob import glob
from collections import defaultdict
from enum import Enum
from py2neo import Database, Graph, Node, Relationship
import argparse
import json

class NodeType(Enum):
    company = 0
    field = 1

class QueryType(Enum):
    company2company = 0,
    company2field = 1,
    field2company = 2,
    field2field = 3

class Neo4jEvaluator:    
    def __init__(self, graph, datapath, top_k, use_gpu=False):
        self.graph = graph

        self.top_k = top_k

        self.evaluate_set = defaultdict(lambda: defaultdict(set))
        self.evaluate_list = defaultdict(lambda: defaultdict(list))

        self.register_dict = self.get_register_dict(datapath)
        
        truth_dir = os.path.join(datapath, 'truth')

        self.load_truth(os.path.join(truth_dir, 'company_related_company.json'), src_type='company', dst_type='company')
        self.load_truth(os.path.join(truth_dir, 'company_related_technology.json'), src_type='company', dst_type='field')
        self.load_truth(os.path.join(truth_dir, 'technology_company.json'), src_type='field', dst_type='company')
        self.load_truth(os.path.join(truth_dir, 'technology_resinst.json'), src_type='field', dst_type='company')    

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

    def get_register_dict(self, datapath):        
        register_path = os.path.join(datapath, 'truth/register/*')

        register_dict = defaultdict(lambda: 'Unknown')

        for filepath in sorted(glob(register_path, recursive=True)):
            if os.path.isdir(filepath): continue
            with open(filepath, encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    register_dict[data['uid']] = data['address']['organisation']

        return register_dict

    def get_eval_field2company(self):
        return [(key, key) for key in self.evaluate_set[QueryType.field2company].keys()]
    
    def get_eval_company2company(self):
        return [(key, self.register_dict[key]) for key in self.evaluate_set[QueryType.company2company].keys()]

    def get_eval_company2field(self):
        return [(key, self.register_dict[key]) for key in self.evaluate_set[QueryType.company2field].keys()]

    def company_to_company(self, zefix_uid):
        query = """MATCH (:Company {{uid: "{zefix_uid}"}})-[:WORKS_ON]->(:Field)<-[:WORKS_ON]-(dst:Company)
                RETURN dst.uid, dst.name, dst.normalized_name
                LIMIT {top_k}""".format(zefix_uid=zefix_uid, top_k=self.top_k)
        return self.graph.run(query).data()

    def field_to_company(self, field_name):
        query = """MATCH (:Field {{label: "{field_name}"}})<-[:WORKS_ON]-(dst:Company)
                RETURN dst.uid, dst.name
                LIMIT {top_k}""".format(field_name=field_name, top_k=self.top_k)
        return self.graph.run(query).data()

    def company_to_field(self, zefix_uid):
        query = """MATCH (:Company {{uid: "{zefix_uid}"}})-[:WORKS_ON]->(field:Field)
                RETURN field.wikidataId, field.label
                LIMIT {top_k}""".format(zefix_uid=zefix_uid, top_k=self.top_k)
        return self.graph.run(query).data()

    def evaluate_node(self, query_type, node_id):
        if query_type == QueryType.company2company:            
            company_list = self.company_to_company(node_id)
            neighbor_list = [company['dst.uid'] for company in company_list]
                  
        elif query_type == QueryType.company2field:
            field_list = self.company_to_field(node_id)
            neighbor_list = [field['field.label'] for field in field_list]

        # elif query_type == QueryType.field2company:
        else:
            company_list = self.field_to_company(node_id)
            neighbor_list = [company['dst.uid'] for company in company_list]
        
        precision = len([neighbor for neighbor in neighbor_list if neighbor in self.evaluate_set[query_type][node_id]]) / self.top_k
        try:
            recall = len([neighbor for neighbor in neighbor_list if neighbor in self.evaluate_set[query_type][node_id]]) / len(self.evaluate_set[query_type][node_id])
        except ZeroDivisionError:
            recall = 0.0

        return precision, recall, neighbor_list, [name for _, name in self.evaluate_list[query_type][node_id][:self.top_k]]
        

    # def evaluate_node(self, ori_node_str, src_type, dst_type, zefix_uid = None):        
    #     ori_node_str = ori_node_str.replace('_', ' ')
    #     node_str = ori_node_str

    #     if src_type.name == 'field':
    #         node_str = term_lookup(ori_node_str)

    #     query_type = QueryType[src_type.name+'2'+dst_type.name]

    #     node_str = node_str.lower()

    #     # if query_type.name == 'field2field':
    #     #     node_id = self.inv_node_dict[node_str]
            
    #     #     num_persona = len(self.persona_map[node_id])

    #     #     queries = np.array([self.persona_emb[str(persona)] for persona in self.persona_map[node_id]])
    #     #     distances, indices = self.db_index[dst_type].search(queries, num_persona * self.top_k) # persona are not connected but we expect to have num_persona duplications of related nodes

    #     #     distances = [dist for batch in distances for dist in batch]
    #     #     indices = [self.inv_persona_map[self.index_map[dst_type][id]] for index in indices for id in index]

    #     #     neighbors = pd.DataFrame(list(zip(indices, distances)))
    #     #     neighbors = neighbors.groupby(0).min().to_dict()[1].items()
    #     #     neighbors = sorted(neighbors, key=lambda x: x[1])                

    #     #     neighbors = [concept_lookup(self.node_dict[neighbor_id]) for neighbor_id, _ in neighbors[:self.top_k]]     

    #     #     return 0 , 0, neighbors, []
    #     if query_type.name == 'company2field':
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

    graph = Graph('bolt://localhost:7687')

    eval = Neo4jEvaluator(graph, datapath=args.datapath, top_k=args.top_k, use_gpu=args.use_gpu)

    evaluation_result = {}

    mAP_dict = {}

    for src_type in NodeType:
        for dst_type in NodeType:
            query_type = QueryType[src_type.name+'2'+dst_type.name]

            if query_type == QueryType.field2field:
                continue

            cum_prec = 0
            matched_count = 0
            query_result = {}
            for node_id in list(eval.evaluate_set[query_type].keys()):
                precision, recall, neighbors, ground_truth = eval.evaluate_node(query_type, node_id)
                if precision > 0.0 or recall > 0.0:
                    query_result[node_id] = {'precision' : precision, 'recall' : recall, 'neighbors' : neighbors, 'ground truth' : ground_truth}
                    matched_count += 1
                    cum_prec += precision
            
            evaluation_result[query_type.name] = query_result
            
            mAP_dict[query_type.name] = {
                'mAP': cum_prec/len(eval.evaluate_set[query_type]),
                'mAP_over_matched': cum_prec/matched_count if matched_count > 0 else 1
            }
    
    json.dump(evaluation_result, open(os.path.join(args.result_path, 'neo4j_evaluation_result.json'), 'w', encoding='utf-8'), indent=2)
    json.dump(mAP_dict, open(os.path.join(args.result_path, 'neo4j_mAP@{k}.txt'.format(k=args.top_k)), 'w', encoding='utf-8'), indent=2)

