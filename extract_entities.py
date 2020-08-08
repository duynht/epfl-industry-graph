from glob import glob
import os
import json
import requests
from itertools import islice

def get_language(text):
    url = 'http://localhost:8090/service/language?text='
    url += text

    lang = "en"

    with requests.get(url) as resp:
        try:
            resp.raise_for_status()
            resp_json = resp.json()
            lang = resp_json["lang"]
        except requests.exceptions.HTTPError as e:
            print(e.__class__, e.response, e.re.request, text)

    return lang

def entity_fishing(text):
    url = 'http://localhost:8090/service/disambiguate'

    query = {"text": text}    

    lang = get_language(text)

    query["language"] = {"lang":lang}        
    
    query["mentions"] = ["ner",  "wikipedia", "wikidata"]

    wikidata_ids = []

    # files = {"query" : query}

    with requests.post(url, json=query) as resp:
        try:
            resp.raise_for_status()
            resp_json = resp.json()
            if 'entities' in resp_json:
                entity_list = resp_json['entities']
                wikidata_ids = [entity['wikidataId'] for entity in entity_list if 'wikidataId' in entity]
        except requests.exceptions.HTTPError as e:
            print(e.__class__, e.response, e.request, query)
    
    return wikidata_ids

def extract_patent(data, ):
    entry = {
        'company_name': data['cleaned_name']
    }

    content = ''

    if not (data['title'] == 'NULL'):
        content += data['title'] + '.'
    if not (data['abstract'] == 'NULL'):
        content += ' ' + data['abstract']

    entry['fields'] = entity_fishing(content)

    return entry

def extract_indeed(data):
    entry = {
        'company_name': data['companyName'].lower()
    }

    content = ''

    if not (data['jobTitle'] == 'NULL'):
        content += data['jobTitle'] + '.'
    if not (data['jobDescription'] == 'NULL'):
        content += ' ' + data['jobDescription']

    entry['fields'] = entity_fishing(content)

    return entry

def extract_all(num_entries):
    output_dir = '../data/extracted/'
    raw_dir = '../data/raw/**'
    log_dir = 'log/'

    batch_size = 500

    entry_list = []
    # task_list = []

    count = 0

    for filepath in sorted(glob(raw_dir, recursive=True), reverse=True):
        print(filepath)
        if os.path.isdir(filepath): continue
        if filepath.split('/')[3] not in {'patent','indeed'}: continue
        if filepath.split('/')[-1] == '_SUCCESS': continue
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.readlines()
            print(len(content))
            for start in range(0, len(content), batch_size):
                if start + batch_size < len(content):
                    end = start + batch_size
                else:
                    end = len(content)

                count += end - start

                line_slice = islice(content, start, end)
                if filepath.split('/')[3] == 'indeed':
                    entry_list = [extract_indeed(json.loads(line)) for line in line_slice]
                elif filepath.split('/')[3] == 'patent':
                    entry_list = [extract_patent(json.loads(line)) for line in line_slice]

                with open(log_dir + filepath.split('/')[3] + '/' + os.path.basename(filepath)+'-log.json', 'w', encoding='utf-8') as f:
                    f.write('CHECKPOINT: '+str(start+1)+'\n')
                    for elem in entry_list:
                        json.dump(elem, f, indent=2)
                        f.write('\n')
                    print('LOGGED!')

                if start == 0:
                    write_mode = 'w'
                else:
                    write_mode = 'a'
                with open(output_dir + filepath.split('/')[3] + '/' + os.path.basename(filepath)+'-extracted.json', write_mode, encoding='utf-8') as f:
                    for elem in entry_list:
                        json.dump(elem, f)
                        f.write('\n')                                                                           

                if count >= num_entries:
                    return

import time
if __name__ == '__main__':
    start = time.perf_counter()
    extract_all(500)    
    stop = time.perf_counter()
    print(stop-start)