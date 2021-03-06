import pandas as pd
# import dask
# import dask.dataframe as dd
# from dask.diagnostics import ProgressBar
from sklearn.metrics import jaccard_score
import glob
import re
import ast

def jaccard(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def get_related(probe, content_list):
    # breakpoint()
    retrieved_list = [(username, jaccard(probe[1], content)) for username, content in content_list if username != probe[0]]
    return sorted(retrieved_list, key=lambda x: x[1], reverse=True)

def precision_at_k(result_list, truth_set, k=40):
    # result_list = probe.retrieved
    # truth_set = probe.crunchbase
    if len(result_list) > k: 
        result_list = result_list[:k]
    
    match_count = 0
    for result in result_list:
        if result[0] in truth_set:
            match_count += 1
    
    return match_count / k

def recall_at_k(result_list, truth_set, k=40):
    # result_list = probe.retrieved
    # truth_set = probe.crunchbase
    if len(result_list) > k: 
        result_list = result_list[:k]
    
    match_count = 0
    for result in result_list:
        if result[0] in truth_set:
            match_count += 1
    
    return match_count / len(truth_set) if len(truth_set) > 0 else match_count

def set_discard(s, val):
    s.discard(val)
    return s

if __name__ == '__main__':
    user_df = pd.read_csv('../data/twitter_swiss_actors/raw/twitter_user.csv', usecols=['id', 'screen_name'], dtype={'id':'string', 'screen_name': 'string'})
    # user_dict = dict(zip(user_df['id'], user_df['screen_name']))
    user_df = user_df.rename(columns = {'id': 'author_id', 'screen_name': 'username'})
    # user_df = user_df[user_df['author_id'].notnull()]

    crunch_df = pd.read_csv('../data/truth/crunchbase/crunchbase.csv', usecols=['Twitter', 'Related'], dtype={'Twitter': 'string'})
    # crunch_df = crunch_df.rename(columns={'Twitter': 'username', 'Organization/Persona Name': 'name', 'Related':'related'})
    # crunch_dict = dict(zip(crunch_df['Twitter'], crunch_df['Related']))
    crunch_df = crunch_df.rename(columns = {'Twitter': 'username', 'Related': 'crunchbase'})
    # crunch_df['crunchbase'].fillna('{}', inplace = True)
    crunch_df = crunch_df[crunch_df['crunchbase'].notnull()]
    crunch_df = crunch_df[crunch_df['username'].notnull()]
    crunch_df['crunchbase'] = crunch_df['crunchbase'].map(ast.literal_eval)
    crunch_df['crunchbase'] = crunch_df['crunchbase'].apply(lambda x: set_discard(x, ''))
   
    # tweet_df = dd.read_csv('twitter_swiss_actors/extracted_document_part*.csv')
    selected_cols = ['author_id', 'entities', 'hashtags']
    # selected_cols = ['author_id', 'body', 'entities']
    tweet_df = pd.concat(
        [pd.read_csv(f, 
            usecols=selected_cols, 
            dtype={'author_id':'string'}) 
        for f in glob.glob('../data/twitter_swiss_actors/extractedv4_document_part*.csv')],
        ignore_index=True)

    # tweet_df = tweet_df[tweet_df['body'].notnull()]

    # tweet_df['hashtags'] = tweet_df.apply(lambda row: re.findall(r"#\w+\s*", row.body), axis=1)

    
    tweet_df = tweet_df[tweet_df['author_id'].notnull()]
    # tweet_df = tweet_df[tweet_df['entities'].notnull() or tweet_df['hashtags'].notnull()]
    tweet_df['entities'].fillna('[]', inplace = True)
    tweet_df['hashtags'].fillna('[]', inplace = True)  
    # tweet_df = tweet_df[tweet_df['author_id'].apply(lambda x: x in user_dict) == True]

    # breakpoint()
    tweet_df['content'] = tweet_df['entities'].apply(lambda x: ast.literal_eval(x)) + tweet_df['hashtags'].apply(lambda x: ast.literal_eval(x))
    tweet_df = tweet_df[['author_id', 'content']]

    tweet_df = tweet_df.groupby('author_id', as_index=False).agg(sum)
    # tweet_df['username'] = tweet_df['author_id'].map(user_dict)
    tweet_df = tweet_df.merge(user_df, on='author_id', how='inner')

    # tweet_df = tweet_df[tweet_df['username'].apply(lambda username: username in crunch_dict) == True]
    ## inner merge will handle this
    content_df = tweet_df[['username','content']]
    content_df['content'] = content_df['content'].map(set)
    # content_df['crunchbase'] = content_df['username'].map(crunch_dict)
    # breakpoint()
    content_df = content_df.merge(crunch_df, on='username', how='inner')

    # with ProgressBar:
    #     content_df = content_df.compute()

    content_rec = content_df[['username', 'content']].to_records(index=False)
    # breakpoint()
    retrieved_dict = {username: get_related((username, content), content_rec) for username, content in content_rec}
    content_df['retrieved'] = content_df['username'].apply(lambda username: retrieved_dict[username])
    content_df['precision@40'] = content_df.apply(lambda row: precision_at_k(row.retrieved, row.crunchbase), axis=1)
    content_df['recall@40'] = content_df.apply(lambda row: recall_at_k(row.retrieved, row.crunchbase), axis=1)

    content_df.to_csv('results/jaccard_result.csv', index=False)

