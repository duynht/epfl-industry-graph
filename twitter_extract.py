import requests
import dask
from dask.diagnostics import ProgressBar
import pandas as pd
import dask.dataframe as dd
import os
import argparse
import re
import json
import sqlalchemy

def entity_fishing(text, lang):
    # url = 'http://nerd.huma-num.fr/nerd/service/disambiguate'
    url = 'http://localhost:8090/service/disambiguate'

    text = re.sub(r'(#|@)\w+\s*', '', text)

    query = {"text": text,
             "language": {"lang": lang},
             "mentions": ["ner", "wikipedia", "wikidata"]}    

    wikidata_ids = []

    try:
      resp = requests.post(url, json=query)
      # resp = requests.post(url, data=query)
      resp.raise_for_status()
      resp_json = resp.json()
      if 'entities' in resp_json:
          entity_list = resp_json['entities']
          wikidata_ids = [entity['wikidataId'] for entity in entity_list if 'wikidataId' in entity]
    except ConnectionResetError  as e:
      print(e)
      print('Trying again...')
      time.sleep(10)
      resp = requests.post(url, json=query)
      resp.raise_for_status()
      resp_json = resp.json()
      if 'entities' in resp_json:
          entity_list = resp_json['entities']
          wikidata_ids = [entity['wikidataId'] for entity in entity_list if 'wikidataId' in entity]
    except Exception as e:
      print()
      print(e)
      print(query)
    
    return wikidata_ids

if __name__ == '__main__':
  dask.config.set(scheduler='threads', num_workers=10)
  # {'id': dtype('O'), 'author_id': dtype('O'), 'author_username': dtype('O'), 'authors': dtype('O'), 'title': dtype('O'), 'body': dtype('O'), 'country': dtype('O'), 'doc_type': dtype('O'), 'language': dtype('O'), 'main_document_id': dtype('O'), 'coordinates': dtype('O'), 'place': dtype('O'), 'user_country': dtype('O'), 'retweet_count': dtype('int64'), 'subject_classes': dtype('O'), 'natural_key': dtype('O'), 'url': dtype('O'), 'document_urls': dtype('O'), 'base_popularity': dtype('int64'), 'popularity': dtype('float64'), 'sentiment': dtype('float64')}
  selected_cols = ['id', 'author_id', 'author_username', 'authors', 'title', 'body', 'country', 'language', 'publishing_date', 'main_document_id', 'user_country']
  connection_string = 
  engine = sqlalchemy.create_engine(connection_string,
                                      pool_size=10,
                                      max_overflow=2,
                                      pool_recycle=300,
                                      pool_pre_ping=True,
                                      pool_use_lifo=True)
  # df = pd.read_sql_table('document', engine, columns=selected_cols, parse_dates={'publishing_date': {'format': '%Y-%m-%d %H:%M:%S'}}, index_col='publishing_date')
  # df = dd.from_pandas(df, npartitions=12)
  # df['entities'] = df.apply(lambda row: entity_fishing(row.body, row.language), axis=1)
  # df['hashtags'] = df.apply(lambda row: re.findall(r"#\w+\s*", row.body), axis=1)
  df = pd.read_sql_table('document', engine)
  df.to_csv('../data/twitter_swiss_actors/document.csv')
  # with ProgressBar():
  #   # df = df.compute()
  #   df.to_csv('../data/twitter_swiss_actors/extractedv4_document_part*.csv', index=False)
