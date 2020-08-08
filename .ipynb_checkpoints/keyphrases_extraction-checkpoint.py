import pke
import ast
from nltk.corpus import stopwords
from glob import glob
import os
import json

stoplist = stopwords.words('english')
window = 3
threshold = 0.8
use_stems = False
pos = {'NOUN'}

all_keyphrases = set()
for filepath in glob('patent/**', recursive=True):
    print(filepath)
    if os.path.isdir(filepath): continue
    with open(filepath,encoding='utf8') as f:
        data_list = []
        for line_num, line in enumerate(f):       
            if not line:
                break
                
            keyphrases_set = set()
            data = ast.literal_eval(line)
            
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
                    keyphrases_set.add(phrase)        
    
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
                    keyphrases_set.add(phrase)
                    all_keyphrases.add(phrase)
        
            entry['fields'] = keyphrases_set
            
        data_list.append(entry)
    with open(filepath+'-extracted', 'w', encoding='utf-8') as f:
        json.dump(data_list, f)

with open('keyphrases.txt', 'w', encoding='utf-8') as f:
    f.write(str(all_keyphrases))