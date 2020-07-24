import argparse
from glob import glob
import os
import json
import aiohttp
import asyncio
from itertools import islice
import uvloop


async def get_language(text):
    url = 'http://localhost:8090/service/language'

    lang = "en"
    # query = {"text": text}

    form = aiohttp.FormData()
    form.add_field('text', text,content_type='multipart/form-data')

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.post(url, data=form) as resp:
            try:
                resp.raise_for_status()
                resp_json = await resp.json()
                lang = resp_json["lang"]
            except aiohttp.client_exceptions.ClientResponseError as e:
                print(e.__class__, e.status, e.message, text)

    return lang



async def entity_fishing(text):
    url = 'http://localhost:8090/service/disambiguate'

    query = {"text": text}    

    lang = await get_language(text)

    query["language"] = {"lang":lang}        
    
    query["mentions"] = ["wikipedia", "wikidata"]

    wikidata_ids = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
        async with session.post(url, json=query) as resp:
            try:
                resp.raise_for_status()
                resp_json = await resp.json()
                if 'entities' in resp_json:
                    entity_list = resp_json['entities']
                    wikidata_ids = [entity['wikidataId'] for entity in entity_list if 'wikidataId' in entity]
            except aiohttp.client_exceptions.ClientResponseError as e:
                print(e.__class__, e.status, e.message, query)
    
    return wikidata_ids

async def extract_patent(data, ):
    entry = {
        'company_name': data['cleaned_name']
    }

    content = ''

    if not (data['title'] == 'NULL'):
        content += data['title'] + '.'
    if not (data['abstract'] == 'NULL'):
        content += ' ' + data['abstract']

    entry['fields'] = await entity_fishing(content)

    return entry

async def extract_indeed(data):
    entry = {
        'company_name': data['companyName'].lower()
    }

    content = ''

    if not (data['jobTitle'] == 'NULL'):
        content += data['jobTitle'] + '.'
    if not (data['jobDescription'] == 'NULL'):
        content += ' ' + data['jobDescription']

    entry['fields'] = await entity_fishing(content)

    return entry

async def async_extract_all(datapath, datasets, num_entries):
    output_dir = os.path.join(datapath, 'extracted/')
    raw_dir = os.path.join(datapath, 'raw/**')
    log_dir = 'log/'

    batch_size = 500

    entry_list = []
    # task_list = []

    count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for ds in datasets:
        p = os.path.join(output_dir, ds)
        if not os.path.exists(p):
            os.makedirs(p)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    for ds in datasets:
        p = os.path.join(log_dir,ds)
        if not os.path.exists(p):
            os.makedirs(p)

    for filepath in sorted(glob(raw_dir, recursive=True), reverse=True):
        print(filepath)
        if os.path.isdir(filepath): continue
        paths = filepath.split('/')
        current_ds = paths[paths.index('raw')+1]
        filename = os.path.basename(filepath)
        if current_ds not in datasets: continue
        if filename == '_SUCCESS': continue
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
                

                if current_ds == 'indeed':
                    entry_list = await asyncio.gather(*[extract_indeed(json.loads(line)) for line in line_slice], return_exceptions=True)
                elif current_ds == 'patent':
                    entry_list = await asyncio.gather(*[extract_patent(json.loads(line)) for line in line_slice],  return_exceptions=True)

                entry_list = [entry for entry in entry_list if not issubclass(type(entry), Exception)]

                with open(os.path.join(log_dir,current_ds,filename+'-log.json'), 'w', encoding='utf-8') as f:
                    f.write('CHECKPOINT: '+str(start+1)+'\n')
                    for elem in entry_list:
                        json.dump(elem, f, indent=2)
                        f.write('\n')
                    print('LOGGED!')

                if start == 0:
                    write_mode = 'w'
                else:
                    write_mode = 'a'
                with open(os.path.join(output_dir,current_ds,filename+'-extracted.json'), write_mode, encoding='utf-8') as f:
                    for elem in entry_list:
                        json.dump(elem, f)
                        f.write('\n')                                                                           

                if num_entries:
                    if count > num_entries:
                        return
                    else:
                        print(count, '/', num_entries)

import time
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_entries', 
                        default=None,
                        type=int,
                        help='number of entries to be extracted (defaulted to extract all)')
    
    parser.add_argument('-d', '--datapath', 
                        default='../data',
                        type=str,
                        help='path to data directory (default is "../data")')

    args = parser.parse_args()

    uvloop.install()
    start = time.perf_counter()
    asyncio.run(async_extract_all(args.datapath, ['indeed'], args.num_entries))
    stop = time.perf_counter()
    print(stop-start)
    print('Done extracting!')
