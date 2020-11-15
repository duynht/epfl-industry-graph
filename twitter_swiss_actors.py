import os
import os.path as osp
import shutil
import glob
import ast
import torch
import pandas as pd
from torch_sparse import coalesce, transpose
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_zip)
from functools import reduce

class TwitterSwissActors(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TwitterSwissActors, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [
            #files in subdir */raw/
            'twitter_user.csv', 'extractedv4_document_part*.csv', 
            #'crunchbase'
        ]

    @property
    def processed_file_names(self):
        return 'twitter_swiss_actors.pt'

    def process(self):
        def flatten_col(df, column):
            '''
            column is a string of the column's name.
            for each value of the column's element (which might be a list),
            duplicate the rest of columns at the corresponding row with the (each) value.
            '''
            column_flat = pd.DataFrame(
                [
                    [i, c_flattened]
                    for i, y in df[column].apply(list).iteritems()
                    for c_flattened in y
                ],
                columns=['I', column]
            )
            column_flat = column_flat.set_index('I')
            return (
                df.drop(column, 1)
                    .merge(column_flat, left_index=True, right_index=True))

        path = osp.join(self.raw_dir, 'twitter_user.csv')
        user_df = pd.read_csv(path, usecols=['id', 'screen_name'], dtype={'id':'string', 'screen_name': 'string'})
        user_df = user_df.rename(columns = {'id': 'author_id', 'screen_name': 'username'})

        path = osp.join(self.raw_dir, 'extractedv4_document_part*.csv')
        selected_cols = ['author_id', 'entities', 'hashtags']
        tweet_df = pd.concat([pd.read_csv(f, usecols=selected_cols, dtype={'author_id':'string'}) for f in glob.glob(path)], ignore_index=True)
        tweet_df = tweet_df[tweet_df['author_id'].notnull()] 
        tweet_df['entities'].fillna('[]', inplace = True)
        tweet_df['hashtags'].fillna('[]', inplace = True)             
        tweet_df['content'] = tweet_df['entities'].apply(lambda x: ast.literal_eval(x)) + tweet_df['hashtags'].apply(lambda x: ast.literal_eval(x))
        tweet_df = tweet_df[['author_id', 'content']]
        tweet_df = tweet_df.groupby('author_id', as_index=False).agg(sum)
        tweet_df['sort_val'] = tweet_df['content'].map(len)
        tweet_df = tweet_df.sort_values(by='sort_val').drop(columns=['sort_val'])
        tweet_df.reset_index(drop=True, inplace=True)
        tweet_df['author_id'] = tweet_df['author_id'].astype('string')
        tweet_df = tweet_df.merge(user_df, on='author_id', how='left')
        tweet_df = tweet_df[['username','content']]
        # tweet_df['content'] = tweet_df['content'].map(set)
        tweet_df['content'] = tweet_df['content'].map(lambda x: reduce(lambda acc, elem: acc+[elem] if not elem in acc else acc, x, []))
        tweet_df.reset_index(inplace=True)
        tweet_df = tweet_df.rename(columns = {'index': 'company_id'})

        #Get company ids
        company = tweet_df[['company_id', 'username']]
        company_index = torch.from_numpy(company['company_id'].values)
        company.to_csv(osp.join(self.processed_dir, 'id_username.csv'))

        # field = pd.DataFrame(list(tweet_df['content'].agg(lambda x: reduce(set.union, x)))])
        # tweet_df.join(pd.concat(
        #     map(lambda x: pd.DataFrame(list(x)), tweet_df['content']),
        #     axis=0
        # ))
        tweet_df = flatten_col(tweet_df, 'content')
        # tweet_df = tweet_df.reset_index(drop=True)
        # tweet_df.reset_index(inplace=True)
        # tweet_df = tweet_df.rename(columns = {'index': 'field_id'})

        #Get field ids
        field = pd.DataFrame({'content': tweet_df['content'].unique()})
        field.reset_index(inplace=True)
        field = field.rename(columns = {'index': 'field_id'})   
        field_index = torch.from_numpy(field['field_id'].values)
        field.to_csv(osp.join(self.processed_dir, 'id_field.csv'))
        
        #Get company<->field connectivity
        tweet_df = tweet_df.merge(field, on='content', how='right')
        company_field = tweet_df[['company_id', 'field_id']]
        company_field = torch.from_numpy(company_field.values)
        company_field = company_field.t().contiguous()

        #This work for the general case
        # M, N = int(company_field[0].max() + 1), int(company_field[1].max() + 1) 
        M, N = company.shape[0], field.shape[0]
        company_field, _ = coalesce(company_field, None, M, N)
        field_company, _ = transpose(company_field, None, M, N)

        # #Get company<->company connectivity
        # path = osp.join(self.raw_dir, 'crunchbase/crunchbase.csv')
        # company_company = pd.read_csv(path, usecols=['Twitter', 'Related'], dtype={'Twitter': str})
        # company_company = company_company..rename(columns = {'Twitter': 'username', 'Related': 'crunchbase'})
        # company_company = company_company.join(pd.concant(
        #     map(lambda x: pd.DataFrame(list(x)), company_company['crunchbase']),
        #     axis=0
        # ))
        
        # company_company = company_company.join(company, on='username', how=)
        # company_company = company_company.rename(columns = {'username': 'src', 'company_id': 'src_id', 'crunchbase': 'username'})
        # company_company = company_company.join(company, on='username', how=)
        # company_company = company_company.rename(columns = {'username': 'dst', 'company_id': 'dst_id'})
        # company_company = company_company[['src_id', 'dst_id']]
        # company_company = torch.from_numpy(company_company.values)
        # company_company = company_company.t().contiguous()

        data = Data(
            edge_index_dict={
                ('company', 'works on', 'field'): company_field,
                ('field', 'attended by', 'company'): field_company,
            },
            y_dict={}, #clusters
            y_index_dict={
                'company': company_index,
                'field': field_index,
            },
            num_nodes_dict={
                'company': company.shape[0],
                'field': field.shape[0],
            }
        )
    
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

