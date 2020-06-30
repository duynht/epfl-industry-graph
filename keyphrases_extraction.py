import pke
import ast
from nltk.corpus import stopwords
from glob import glob
import os
import json
import requests
from bs4 import BeautifulSoup as bs


def get_entity_str(id, phrase):
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetentities',
        'format': 'json',
        'ids': id
    }
    r = requests.get(url, params=params)

    entity_str = ''
    
    try:
        if not (r.status_code == 200):
            r = requests.get(url, params=params)
            if(r.status_code == 200):
                entity_str = r.json()['entities'][id]['labels']['en']['value']
        else:
            entity_str = r.json()['entities'][id]['labels']['en']['value']
    except KeyError:
        pass
    
    return entity_str

def get_page_title(url):
    r = requests.get(url)
    soup = bs(r.content, 'lxml')
    return soup.select_one('title').text

def falcon_call(text):
    url = 'https://labs.tib.eu/falcon/falcon2/api?mode=long'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    entities=[]
    payload = '{"text":"'+text+'"}'
    r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
    if r.status_code == 200:
        response=r.json()
        for result in response['entities_wikidata']:
            entities.append(result[0])
    else:
        r = requests.post(url, data=payload.encode('utf-8'), headers=headers)
        if r.status_code == 200:
            response=r.json()
            for result in response['entities_wikidata']:
                entities.append(result[0])

    entities = [entity[1:-1].split('/')[-1] for entity in entities]
    return entities

if __name__ == '__main__':
    stoplist = stopwords.words('english')
    window = 3
    threshold = 0.8
    use_stems = False
    pos = {'NOUN'}

    extracted_dir = 'data/raw/patent-extracted/'
    raw_dir = 'data/raw/patent/**'
    log_dir = 'log/patent/'

    all_keyphrases = set()
    for filepath in sorted(glob(raw_dir, recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        with open(filepath, 'r', encoding='utf-8') as f:
            data_list = []
            for line_num, line in enumerate(f):   
                print('PROCESSED ENTRIES: ', line_num)
                keyphrases_set = set()
                # data = ast.literal_eval(line)
                data = json.loads(line)
                
                entry = {}
                entry['company_name'] = data['cleaned_name']
                entry['country'] = data['application_country']
        
                if not (data['title'] == 'NULL'):
                    extractor = pke.unsupervised.MultipartiteRank()
                    extractor.load_document(input=data['title'], language='en')
                    extractor.candidate_selection(pos=pos, stoplist=stoplist)
                    if(len(extractor.candidates) > 1):
                        extractor.candidate_weighting()
                    keyphrases = extractor.get_n_best(n=10)
                    
                    for phrase,_ in keyphrases:
                        entities = falcon_call(phrase)
                        for entity in entities:
                            entity_str = get_entity_str(entity, phrase).lower()
                            if not (entity_str == ''):
                                keyphrases_set.add(entity_str)
                                all_keyphrases.add(entity_str)        
        
                if not (data['abstract'] == 'NULL'):
                    # extractor = pke.unsupervised.YAKE()
                    # extractor.load_document(input=data['abstract'], language='en')
                    # extractor.candidate_selection(n=2, stoplist=stoplist)
                    # extractor.candidate_weighting(window=window, use_stems=use_stems)
                    # keyphrases = extractor.get_n_best(n=5, threshold=threshold)

                    extractor = pke.unsupervised.MultipartiteRank()
                    extractor.load_document(input=data['abstract'], language='en')
                    extractor.candidate_selection(pos=pos, stoplist=stoplist)
                    if (len(extractor.candidates) > 1):
                        extractor.candidate_weighting()
                    keyphrases = extractor.get_n_best(n=10)

                    for phrase,_ in keyphrases:
                        entities = falcon_call(phrase)
                        for entity in entities:
                            entity_str = get_entity_str(entity, phrase).lower()
                            if not (entity_str == ''):
                                keyphrases_set.add(entity_str)
                                all_keyphrases.add(entity_str)     
            
                entry['fields'] = list(keyphrases_set)
                data_list.append(entry)

                if (line_num + 1) % 1000 == 0 or line_num == 0:                
                    write_mode = 'w' if line_num == 0 else 'a'
                    with open(log_dir + 'patent-entries-extracted.json', write_mode, encoding='utf-8') as f:
                        for elem in data_list:
                            json.dump(elem, f, indent=2)
                            f.write('\n')
                        f.write('CHECKPOINT: '+str(line_num+1)+'\n')
                                    
                    with open(log_dir + 'patent-keyphrases.json', 'w', encoding='utf-8') as f:
                        json.dump(list(all_keyphrases), f, indent=2)
                    
                    with open(extracted_dir + os.path.basename(filepath)+'-extracted.json', write_mode, encoding='utf-8') as f:
                        for elem in data_list:
                            json.dump(elem, f)
                            f.write('\n')
                    
                    data_list.clear()
                    if (line_num + 1 == 10000):
                        break

        with open(extracted_dir + os.path.basename(filepath)+'-extracted.json', 'a', encoding='utf-8') as f:
            for elem in data_list:
                json.dump(data_list, f)
                f.write('\n')

    with open(log_dir + 'patent-keyphrases.json', 'w', encoding='utf-8') as f:
        json.dump(list(all_keyphrases), f, indent=2)
