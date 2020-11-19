import dask
from dask.diagnostics import ProgressBar
import pandas as pd
import dask.dataframe as dd
import glob
from urllib.parse import urlparse
import argparse

def match_cat(row, ref_list):
  return {ref[0] for ref in ref_list if row[0] != ref[0] and not row[2].isdisjoint(ref[2])}


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--input', type=str, required=True)
  args = parser.parse_args()

  filepath = args.input

  # dask.config.set(scheduler='processes')
  
  # selected_cols = ['Twitter', 'Organization/Person Name', 'Categories']
  selected_cols = ['twitter_url', 'name', 'category_list']
  
  # df = dd.read_csv('organizations-or-people-01-11-2019-part*.csv')
  # df = pd.concat([pd.read_csv(f) for f in glob.glob('organizations-or-people-01-11-2019-part*.csv')], ignore_index=True)

  df = pd.read_csv(filepath)
  df = df[selected_cols]
  df = df.rename(columns={'twitter_url': 'Twitter', 'name': 'Organization/Person Name', 'category_list': 'Categories'})
  
  df = df[df['Twitter'].notnull()]
  df = df[df['Categories'].notnull()]
  # df['Twitter'] = df['Twitter'].apply(lambda row: row.split('/')[-1])
  df['Twitter'] = df['Twitter'].apply(lambda row: urlparse(row).path.split('/')[-1])
  df = df[df['Twitter'].notnull()]
  df['Categories'] = df['Categories'].apply(lambda row: set(row.split(',')))

  df['Related'] = [{ref for _,ref,_,ref_cat_list in df.itertuples() if ref != username and not cat_list.isdisjoint(ref_cat_list)} for _,username, _, cat_list in df.itertuples()]
  df.to_csv('nda_crunchbase.csv', index=False)