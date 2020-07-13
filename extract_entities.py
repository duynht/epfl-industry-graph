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
        async with session.post(url, json=query) as resp:
            try:
                resp.raise_for_status()
                anno = await resp.json()
                if 'entities' in anno:
                    entity_list = anno['entities']
                    wikidata_ids = [entity['wikidataId'] for entity in entity_list if 'wikidataId' in entity]
            except aiohttp.client_exceptions.ClientResponseError as e:
                print(e.__class__, e.status, e.message, query)
    
    return wikidata_ids

    


    # try:
    #     response = await session.post(url, files=payload)
    #     if response.status_code == 200:
    #         entities = response.json()['entities']
    #         for entity in entities:
    #             wikidata_ids.append(entity['wikidataId'])
    # except Exception as err:
    #     pass

    # return wikidata_ids 

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

# async def extract_batch(batch, session):
#     if filepath.split('/')[3] == 'patent':
#         entry_list = await asyncio.gather(*[extract_patent(json.loads(line), session) for line in line_slices])
#     elif filepath.split('/')[3] == 'indeed':
#         entry_list = await asyncio.gather(*[extract_indeed(json.loads(line), session) for line in line_slices])


# async def extract_file(filepath, event_loop):
#     entries = []
#     with open(filepath, 'r', encoding='utf-8') as f:
#             if filepath.split('/')[3] == 'patent':
#                 for line in f:
                
#                 entry_list = await asyncio.gather(*[extract_patent(json.loads(line), session) for line in f])
#             elif filepath.split('/')[3] == 'indeed':
#                 entry_list = await asyncio.gather(*[extract_indeed(json.loads(line), session) for line in f])

async def async_extract_all(num_entries):
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

                if count >= num_entries:
                    return
    #         for line in f:
    #             if filepath.split('/')[3] == 'indeed':
    #                 task_list.append(loop.create_task(extract_indeed(json.loads(line))))
    #             elif filepath.split('/')[3] == 'patent':
    #                 task_list.append(loop.create_task(extract_patent(json.loads(line))))
            

    # for start in range(0, len(tasks), batch_size):
    #     if batch + batch_size < len(tasks):
    #         end = batch + batch_size
    #     else:
    #         end = len(tasks)
    #     task_slice = task_list[start : end]
    #     entry_list = []
    #     for task in task_slice:
    #         entry_list.append(await task)

    #     with open(log_dir + os.path.basename(filepath)+'-log.json', 'w', encoding='utf-8') as f:
    #         f.write('CHECKPOINT: '+str(start+1)+'\n')
    #         for elem in entry_list:
    #             json.dump(elem, f, indent=2)
    #             f.write('\n')

    #     if start == 0:
    #         write_mode = 'w'
    #     else:
    #         write_mode = 'a'
    #     with open(output_dir + os.path.basename(filepath)+'-extracted.json', write_mode, encoding='utf-8') as f:
    #         for elem in entry_list:
    #             json.dump(elem, f)
    #             f.write('\n')                       

if __name__ == '__main__':
    asyncio.run(async_extract_all(10000))

    # extracted_dir = 'data/raw/patent-extracted/'
    # raw_dir = 'data/raw/'
    # log_dir = 'log/patent/'

    # all_keyphrases = set()
    # async with aiohttp.ClientSession() as session:
    # for filepath in sorted(glob(raw_dir, recursive=True)):
    #     print(filepath)
    #     if os.path.isdir(filepath): continue
    #     with open(filepath, 'r', encoding='utf-8') as f:
    #         # TODO: If patent
        
    #         data_list = []
            
    #         for line_num, line in enumerate(f):   
    #             print('PROCESSED ENTRIES: ', line_num)
    #             keyphrases_set = set()
    #             data = json.loads(line)
                
    #             entry = {}
    #             entry['company_name'] = data['cleaned_name']
    #             entry['country'] = data['application_country']
        
    #             if not (data['title'] == 'NULL'):
    #                 extractor = pke.unsupervised.MultipartiteRank()
    #                 extractor.load_document(input=data['title'], language='en')
    #                 extractor.candidate_selection(pos=pos, stoplist=stoplist)
    #                 if(len(extractor.candidates) > 1):
    #                     extractor.candidate_weighting()
    #                 keyphrases = extractor.get_n_best(n=10)
                    
    #                 for phrase,_ in keyphrases:
    #                     entities = falcon_call(phrase)
    #                     for entity in entities:
    #                         entity_str = get_entity_str(entity, phrase).lower()
    #                         if not (entity_str == ''):
    #                             keyphrases_set.add(entity_str)
    #                             all_keyphrases.add(entity_str)        
        
    #             if not (data['abstract'] == 'NULL'):
    #                 # extractor = pke.unsupervised.YAKE()
    #                 # extractor.load_document(input=data['abstract'], language='en')
    #                 # extractor.candidate_selection(n=2, stoplist=stoplist)
    #                 # extractor.candidate_weighting(window=window, use_stems=use_stems)
    #                 # keyphrases = extractor.get_n_best(n=5, threshold=threshold)

    #                 extractor = pke.unsupervised.MultipartiteRank()
    #                 extractor.load_document(input=data['abstract'], language='en')
    #                 extractor.candidate_selection(pos=pos, stoplist=stoplist)
    #                 if (len(extractor.candidates) > 1):
    #                     extractor.candidate_weighting()
    #                 keyphrases = extractor.get_n_best(n=10)

    #                 for phrase,_ in keyphrases:
    #                     entities = falcon_call(phrase)
    #                     for entity in entities:
    #                         entity_str = get_entity_str(entity, phrase).lower()
    #                         if not (entity_str == ''):
    #                             keyphrases_set.add(entity_str)
    #                             all_keyphrases.add(entity_str)     
            
    #             entry['fields'] = list(keyphrases_set)
    #             data_list.append(entry)

    #             if (line_num + 1) % 1000 == 0 or line_num == 0:                
    #                 write_mode = 'w' if line_num == 0 else 'a'
    #                 with open(log_dir + 'patent-entries-extracted.json', write_mode, encoding='utf-8') as f:
    #                     for elem in data_list:
    #                         json.dump(elem, f, indent=2)
    #                         f.write('\n')
    #                     f.write('CHECKPOINT: '+str(line_num+1)+'\n')
                                    
    #                 with open(log_dir + 'patent-keyphrases.json', 'w', encoding='utf-8') as f:
    #                     json.dump(list(all_keyphrases), f, indent=2)
                    
    #                 with open(extracted_dir + os.path.basename(filepath)+'-extracted.json', write_mode, encoding='utf-8') as f:
    #                     for elem in data_list:
    #                         json.dump(elem, f)
    #                         f.write('\n')
                    
    #                 data_list.clear()
    #                 if (line_num + 1 == 10000):
    #                     break

    #     with open(extracted_dir + os.path.basename(filepath)+'-extracted.json', 'a', encoding='utf-8') as f:
    #         for elem in data_list:
    #             json.dump(data_list, f)
    #             f.write('\n')

    # with open(log_dir + 'patent-keyphrases.json', 'w', encoding='utf-8') as f:
    #     json.dump(list(all_keyphrases), f, indent=2)
