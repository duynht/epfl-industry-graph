from py2neo import Database, Graph, Node, Relationship
import argparse
import pickle
import json
import os
from glob import glob
import csv
from collections import defaultdict
import requests
import wptools

# def concept_lookup(id):
#     concept = 'Unknown'
#     url = 'http://localhost:8090/service/kb/concept/'+id
#     with requests.get(url) as resp:
#         try:
#             resp.raise_for_status()
#             try:
#                 anno = resp.json()
#                 try:
#                     concept = anno['preferredTerm']
#                 except KeyError as e:
#                     print(e.__class__, e, anno)
#             except json.decoder.JSONDecodeError as e:
#                 print(e.__class__, e, resp.text)
#         except requests.exceptions.HTTPError as e:
#             print(e.__class__, e, e.response, e.re.request, text)
    
#     return concept.lower()

async def concept_lookup(id):
    url = 'http://localhost:8090/service/kb/concept/'+id
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.get(url) as resp:
            try:
                resp.raise_for_status()
                try:
                    resp_json = await resp.json()
                    try:
                        concept = resp_json['preferredTerm']
                    except KeyError as e:
                        print(e.__class__, e, resp_json)
                except json.decoder.JSONDecodeError as e:
                    print(e.__class__, e, resp.text)
            except aiohttp.client_exceptions.ClientResponseError as e:
                print(e.__class__, e, id)

    return concept.lower() if concept else 'Unknown'

def get_wikidataLabel(id):
    page = wptools.page(wikibase=id, silent=True).get_wikidata()
    label = page.data['label']
    return label.lower() if label else 'Unknown'

# def get_qwkidataLabel(id):
#     entity_dict = qwikid


def get_register_dict(datapath):        
    register_path = os.path.join(datapath, 'truth/register/*')

    register_dict = defaultdict(lambda: 'Unknown')

    for filepath in sorted(glob(register_path, recursive=True)):
        if os.path.isdir(filepath): continue
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                register_dict[data['address']['organisation'].lower()] = data['uid'] 

    return register_dict


def _parse_graph(datapath, dataset, unwanted_set, graph):
    register_dict = get_register_dict(datapath)
    extracted_path = os.path.join(datapath, 'extracted/**')

    for filepath in sorted(glob(extracted_path, recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        with open(filepath,encoding='utf-8') as f:
            for line in enumerate(f):
                try:        
                    data = json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    print(e)
                    print(line)
                    # raise

                if data['fields']:
                    company_node = Node("Company", 
                                        name=data['company_name'],
                                        uid=register_dict[data['company_name']])

                    graph.begin(autocommit=True).merge(company_node,"Company","name")

                    for field in data['fields']:
                        field_node = Node("Field",
                                          wikidataId=field,
                                          name=get_wikidataLabel(field))
                        graph.begin(autocommit=True).merge(field_node, "Field", "wikidataId")
                        graph.begin(autocommit=True).merge(Relationship(company_node, "WORKS_ON", field_node))
    return

def parse_graph(datapath, datasets, unwanted_set, graph):
    register_dict = get_register_dict(datapath)
    extracted_path = os.path.join(datapath, 'entities/**')

    for filepath in sorted(glob(extracted_path, recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        current_ds = filepath.split('/')[-2]

        with open(filepath,encoding='utf-8') as f:
            for line in f:       
                try:        
                    data = json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    print(e)
                    print(line)
                    # raise
                    continue

                field_list = []
                try:
                    for field in json.loads(data['result'])['entities']:
                        if (field['confidence_score'] > 0.5) and ('type' not in field.keys() or field['type'] not in unwanted_set) and ('wikidataId' in field.keys()):
                            field_list.append({'wikidataId': field['wikidataId'], 
                                            'pageId': field['wikipediaExternalRef'],
                                            # 'name': concept_lookup(field['wikidataId'])})
                                            'name': get_wikidataLabel(field['wikidataId'])})    
                except Exception as e:
                    continue

                if current_ds == 'indeed':
                    company_name = data['companyName']
                else:
                    company_name = data['company_name']

                company_node = Node("Company", 
                                    name=company_name,
                                    normalized_name=company_name.lower(),
                                    uid=register_dict[company_name.lower()])
                
                graph.begin(autocommit=True).merge(company_node, "Company", "name")

                for field in field_list:
                    field_node = Node("Field", 
                                        wikidataId=field['wikidataId'],
                                        pageId=field['pageId'],
                                        name=field['name'])
                    graph.begin(autocommit=True).merge(field_node, "Field", "wikidataId")
                    graph.begin(autocommit=True).merge(Relationship(company_node, "WORKS_ON", field_node))

    return


def company_to_company(company_name, graph):
    return graph.run("MATCH (company:Company {name: '" + company_name +"'})-[:WORKS_ON]->(field:Field)<-[:WORKS_ON]-(company:Company)").data()

def field_to_company(field_name, graph):
    return graph.run("MATCH (field:Field {name: '" + field_name +"'})<-[:WORKS_ON]-(company:Company)").data()

def company_to_field(company_name, graph):
    return graph.run("MATCH (company:Company {name: '" + company_name +"'})-[:WORKS_ON]->(field:Field)").data()

if __name__ == '__main__':
    unwanted_set = {'ARTIFACT', 
                    'ACRONYM',
                    'ANIMAL',
                    'ARTIFACT',
                    'AWARD',
                    'BUSINESS',
                    'EVENT',
                    'IDENTIFIER',
                    'INSTALLATION',
                    'INSTITUTION',
                    'LOCATION',
                    'MEASURE',
                    'MEDIA',
                    'NATIONAL',
                    'ORGANISATION', 
                    'PERIOD',
                    'PERSON',
                    'PERSON_TYPE',
                    'PLANT',
                    'SPORT_TEAM',
                    'SUBSTANCE',
                    'TITLE',
                    'WEBSITE'}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datapath', 
                        default='../data',
                        type=str,
                        help='path to data directory (default is "../data")')

    args = parser.parse_args()
    graph = Graph('bolt://localhost:7687')
    graph.delete_all()
    parse_graph(args.datapath, {'indeed','patent'}, unwanted_set, graph)
    import pdb; pdb.set_trace()


    
    
