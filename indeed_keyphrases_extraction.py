import pke
import ast
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words
from glob import glob
import os
import json
import re

# TODO: YAKE bug

if __name__ == "__main__":
    stoplist = stopwords.words('english') + stopwords.words('german')
    window = 3
    threshold = 0.8
    use_stems = False
    pos = {'NOUN'}

    all_keyphrases = set()
    for filepath in sorted(glob('indeed/**', recursive=True)):
        print(filepath)
        if os.path.isdir(filepath): continue
        with open(filepath,encoding='utf8') as f:
            data_list = []
            for line_num, line in enumerate(f):       
                if not line:
                    break
                    
                keyphrases_set = set()
                data = ast.literal_eval(line)

                data['jobTitle'] = " ".join(w for w in nltk.wordpunct_tokenize(data['jobTitle']) \
                                                if w.lower() in words.words())
                
                entry = {}
                entry['company_name'] = re.sub(r'[^a-zA-Z\d]','', data['companyName'].strip().lower())
        
                if not (data['jobTitle'] == 'NULL'):
                    extractor = pke.unsupervised.YAKE()
                    extractor.load_document(input=data['jobTitle'], language='en')
                    extractor.candidate_selection(n=1, stoplist=stoplist)
                    if (len(extractor.candidates) > 1):
                        extractor.candidate_weighting(window=window, use_stems=use_stems)
                    keyphrases = extractor.get_n_best(n=10, threshold=threshold)

                    # extractor = pke.unsupervised.MultipartiteRank()
                    # extractor.load_document(input=data['jobTitle'], language='en')
                    # extractor.candidate_selection(pos=pos, stoplist=stoplist)
                    # if(len(extractor.candidates) > 1):
                    #     extractor.candidate_weighting()
                    # keyphrases = extractor.get_n_best(n=10)
                    
                    for phrase,_ in keyphrases:
                        keyphrases_set.add(phrase)        

                data['jobDescription'] = " ".join(w for w in nltk.wordpunct_tokenize(data['jobDescription']) \
                                                    if w.lower() in words.words())
        
                if not (data['jobDescription'] == 'NULL'):
                    extractor = pke.unsupervised.YAKE()
                    extractor.load_document(input=data['jobDescription'], language='en')
                    extractor.candidate_selection(n=1, stoplist=stoplist)
                    if (len(extractor.candidates) > 1):
                        extractor.candidate_weighting(window=window, use_stems=use_stems)
                    keyphrases = extractor.get_n_best(n=10, threshold=threshold)

                    # extractor = pke.unsupervised.MultipartiteRank()
                    # extractor.load_document(input=data['jobDescription'], language='en')
                    # extractor.candidate_selection(pos=pos, stoplist=stoplist)
                    # if (len(extractor.candidates) > 1):
                    #     extractor.candidate_weighting()
                    # keyphrases = extractor.get_n_best(n=10)

                    for phrase,_ in keyphrases:
                        keyphrases_set.add(phrase)
                        all_keyphrases.add(phrase)
            
                entry['fields'] = list(keyphrases_set)
                data_list.append(entry)

                if (line_num % 1000 == 0):                
                    write_mode = 'w' if line_num == 0 else 'a'
                    with open('./log/indeed/indeed-entries-extracted.json', write_mode, encoding='utf-8') as f:
                        f.write(str(line_num)+'\n')
                        for elem in data_list:
                            json.dump(elem, f, indent=2)
                            f.write('\n')
                                    
                    with open('./log/indeed/indeed-keyphrases.json', 'w', encoding='utf-8') as f:
                        json.dump(list(all_keyphrases), f, indent=2)
                    
                    with open(filepath+'-extracted.json', write_mode, encoding='utf-8') as f:
                        for elem in data_list:
                            json.dump(elem, f)
                            f.write('\n')
                    
                    data_list.clear()

        with open(filepath+'-extracted.json', 'a', encoding='utf-8') as f:
            for elem in data_list:
                json.dump(data_list, f)
                f.write('\n')

    with open('./log/indeed/indeed-keyphrases.json', 'w', encoding='utf-8') as f:
        json.dump(list(all_keyphrases), f, indent=2)