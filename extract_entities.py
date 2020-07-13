import pke
import ast
from glob import glob
import os
import json
import requests
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
import aiohttp
import asyncio
from itertools import islice
import uvloop
import pickle


async def entity_fishing(text, is_title=False):
    url = 'http://localhost:8090/service/disambiguate'

    if is_title:
        query = { "shortText": text}
    else:
        query = {"text": text}

    query["language"] = {
        "lang": "en",
        "lang": "de",
        "lang": "fr"
    }    
    
    query["mentions"] = ["ner",  "wikipedia", "wikidata"]

    wikidata_ids = []

    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=query, timeout=60) as resp:
            try:
                resp.raise_for_status()
                anno = await resp.json()
                if 'entities' in anno:
                    entity_list = anno['entities']
                    wikidata_ids = [entity['wikidataId'] for entity in entity_list if 'wikidataId' in entity]
            except aiohttp.client_exceptions.ClientResponseError as e:
                print(e.__class__, e.status, e.message, query)
    
    return wikidata_ids

async def extract_patent(data, ):
    entry = {
        'company_name': data['cleaned_name']
    }

    title_fields = []
    abstract_fields = []
    if not (data['title'] == 'NULL'):
        title_fields = await entity_fishing(data['title'], is_title=True)
    if not (data['abstract'] == 'NULL'):
        abstract_fields = await entity_fishing(data['abstract'])

    entry['fields'] = title_fields + abstract_fields

    return entry

async def extract_indeed(data):
    entry = {
        'company_name': data['companyName'].lower()
    }

    jt_fields = []
    jd_fields = []
    if not (data['jobTitle'] == 'NULL'):
        jt_fields = await entity_fishing(data['jobTitle'], is_title=True)
    if not (data['jobDescription'] == 'NULL'):
        jd_fields = await entity_fishing(data['jobDescription'])

    entry['fields'] = jt_fields + jd_fields

    return entry

async def async_extract_all():
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
                    entry_list = await asyncio.gather(*[extract_indeed(json.loads(line)) for line in line_slice])
                elif filepath.split('/')[3] == 'patent':
                    entry_list = await asyncio.gather(*[extract_patent(json.loads(line)) for line in line_slice])

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
             

if __name__ == '__main__':
    asyncio.run(async_extract_all())
