import io 
import pandas as pd
import dask.dataframe as dd
import glob
from urllib.parse import urlparse
import argparse
import csv

def match_cat(row, ref_list):
  return {ref[0] for ref in ref_list if row[0] != ref[0] and not row[2].isdisjoint(ref[2])}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  parser.add_argument('-o', '--output', type=str, required=True)
  parser.add_argument('-s', '--sep', type=str, default=',')
  args = parser.parse_args()

  input_filepath = args.input
  output_filepath = args.output

  # dask.config.set(scheduler='processes')
  
  # selected_cols = ['Twitter', 'Organization/Person Name', 'Categories']
  selected_cols = ['twitter_url', 'name', 'category_list']
  
  # df = dd.read_csv('organizations-or-people-01-11-2019-part*.csv')
  # df = pd.concat([pd.read_csv(f) for f in glob.glob('organizations-or-people-01-11-2019-part*.csv')], ignore_index=True)
  # with open(input_filepath) as f:
  #   file = io.StringIO(f.read().replace('"',"'"))
    
  df = pd.read_csv(input_filepath, sep=args.sep, skiprows=[151142, 510834])
  # df = pd.read_csv(input_filepath, quotechar='"', delim_whitespace=True)
  df = df[selected_cols]
  df = df.rename(columns={'twitter_url': 'Twitter', 'name': 'Organization/Person Name', 'category_list': 'Categories'})
  
  df = df[df['Twitter'].notnull()]
  df = df[df['Categories'].notnull()]
  # df['Twitter'] = df['Twitter'].apply(lambda row: row.split('/')[-1])
  df['Twitter'] = df['Twitter'].apply(lambda row: urlparse(row).path.split('/')[-1])
  df = df[df['Twitter'].notnull()]
  df['Categories'] = df['Categories'].apply(lambda row: set(row.split(',')))

  # df['Related'] = [{ref for _,ref,_,ref_cat_list in df.itertuples() if ref != username and not cat_list.isdisjoint(ref_cat_list)} for _,username, _, cat_list in df.itertuples()]
  df['Related'] = df.apply(lambda row: match_cat(row, df.values.tolist()), axis=1)
  df.to_csv(output_filepath, index=False)
