import pandas as pd
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.metrics import jaccard_score
import glob

def get_related(probe, content_list):
    retrieved_list = [(username, jaccard_score(probe[1], content)) for username, content in content_list if username != probe[0]]
    return sorted(retrieved_list, lambda x: x[1], reverse=True)

def precision_at_k(result_list, truth_set, k=40):
    if len(result_list) > k: 
        result_list = result_list[:k]
    
    match_count = 0
    for result in result_list:
        if result[0] in truth_set:
            match_count += 1
    
    return match_count / k

def recall_at_k(result_list, truth_set, k=40):
    if len(result_list) > k: 
        result_list = result_list[:k]
    
    match_count = 0
    for result in result_list:
        if result[0] in truth_set:
            match_count += 1
    
    return match_count / len(truth_set)



if __name__ == '__main__':
    user_df = pd.read_csv('twitter_swiss_actors/twitter_user.csv')
    user_dict = dict(zip(user_df['id'], user_df['screen_name']))

    crunch_df = pd.read_csv('crunchbase/crunchbase.csv')
    # crunch_df = crunch_df.rename(columns={'Twitter': 'username', 'Organization/Persona Name': 'name', 'Related':'related'})
    crunch_dict = dict(zip(crunch_df['Twitter'], crunch_df['Related']))

    tweet_df = dd.read_csv('gdrive/extracted_document_part*.csv')
    # tweet_df = pd.concat([pd.read_csv(f) for f in glob.glob('twitter_swiss_actors/extracted_document_part*.csv')], ignore_index=True)
    tweet_df = tweet_df[tweet_df['author_id'].notnull()]
    
    tweet_df['content'] = tweet_df.apply(lambda row: row.entities + row.hashtags, axis=1)
    tweet_df = tweet_df[['author_id', 'content']]
    tweet_df['username'] = tweet_df.apply(lambda row: user_dict[row.author_id], axis=1)
    tweet_df = tweet_df[tweet_df['username'].apply(lambda username: username in crunch_dict)]
    tweet_df = tweet_df[['username','content']]

    content_df = tweet_df.groupby('username', as_index=False).aggregate({'content': 'sum'})
    content_df['content'] = content_df['content'].apply(set())

    with ProgressBar:
        content_df = content_df.compute()

    content_rec = content_df.to_records(index=False)

    retrieved_dict = {username: get_related(content, content_rec) for username, content in content_rec}
    content_df['retrieved'] = content_df['username'].apply(lambda username: retrieved_dict['username'])
    content_df['precision@40'] = content_df.apply(lambda row: precision_at_k(row.retrieved, crunch_dict[row.username]), axis=1)
    content_df['recall@40'] = content_df.apply(lambda row: recall_at_k(row.retrieved, crunch_dict[row.username]), axis=1)

    content_df.to_csv('jaccard_result.csv', index=False)

