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
import dask
from dask.diagnostics import ProgressBar
import pandas as pd

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

def get_wikidataLabel(id):
    try:
        page = wptools.page(wikibase=id, silent=True).get_wikidata()
        label = page.data['label']
        return label.lower()
    except Exception as e:
        print(e)
        return 'Unknown'

def process(current_ds, line, register_dict):
    try:
        data = json.loads(line)


        field_dict = {}
        for field in json.loads(data['result'])['entities']:
            if (field['confidence_score'] > 0.5) and ('type' not in field.keys() or field['type'] not in unwanted_set) and ('wikidataId' in field.keys()):
                field_dict[field['wikidataId']] = {'pageId': field['wikipediaExternalRef']}
                                # 'name': concept_lookup(field['wikidataId'])})
                                # 'name': dask.delayed(get_wikidataLabel)(field['wikidataId'])})

        if current_ds == 'indeed':
            company_name = data['companyName']
        else:
            company_name = data['company_name']


        company = {'uid': register_dict[company_name.lower()],
                    'name': company_name,
                    'normalized_name': company_name.lower()}

        rel_list = [(company['uid'], field_id) for field_id in field_dict.keys()]

        return company, field_dict, rel_list
    except Exception as e:
        print(e)
        return None, None, None

def add_labels(field_dict):   
    label_list = [dask.delayed(get_wikidataLabel)(field_id) for field_id in field_dict.keys()]

    print('Retrieving field labels...')
    with ProgressBar():
        label_list = dask.compute(*label_list, scheduler='threads')

    for i, (field_id, field) in enumerate(field_dict.items()):
        field.update({'label': label_list[i]})

    return field_dict

def parse_entities(datapath, datasets, unwanted_set):
    register_dict = get_register_dict(datapath)
    extracted_path = os.path.join(datapath, 'entities/**')

    delayed_list = []

    for filepath in sorted(glob(extracted_path, recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        current_ds = filepath.split('/')[-2]

        with open(filepath,encoding='utf-8') as f:
            for line in f:
                delayed_list.append(dask.delayed(process)(current_ds, line, register_dict))

    print('Parsing files...')
    with ProgressBar():
        delayed_list = dask.compute(*delayed_list, scheduler='threads')
    delayed_list = [result for result in delayed_list if not None in result]

    company_list, field_ldict, rel_llist = zip(*delayed_list)

    company_dict = {company['uid']:company for company in company_list if not company['uid'] == 'Unknown'}
    field_dict = {field_id:field for fdict in field_ldict for field_id, field in fdict.items()}
    rel_list = [rel for rel_list in rel_llist for rel in rel_list if not 'Unknown' in rel]
    
    return company_dict, field_dict, rel_list

def parse_to_csv(datapath, datasets, unwanted_set):
    output_dir = os.path.join(datapath, 'csv')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    company_dict, field_dict, rel_list = parse_entities(datapath, datasets, unwanted_set)
    
    pd.DataFrame([{'uid': uid, 'name': company['name'], 'normalized_name': company['normalized_name']} for uid, company in company_dict.items()]).to_csv(os.path.join(output_dir, 'companies.csv'), encoding='utf-8')

    pd.DataFrame(rel_list, columns=['company', 'field']).to_csv(os.path.join(output_dir, 'relationships.csv'), encoding='utf-8')

    field_dict = add_labels(field_dict)
    pd.DataFrame([{'wikidataId': wikidataId, 'pageId': field['pageId'], 'label': field['label']} for wikidataId, field in field_dict.items()).to_csv(os.path.join(output_dir,'fields.csv'), encoding='utf-8')
    
    return

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
                        default='../../data',
                        type=str,
                        help='path to data directory (default is "../data")')

    args = parser.parse_args()
    parse_to_csv(args.datapath, {'indeed','patent'}, unwanted_set)