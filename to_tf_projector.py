from gensim.models import KeyedVectors
import pandas as pd
import os.path as osp
import numpy as np

if __name__ == "__main__":
    path = '/home/tuanduy/data/twitter_swiss_actors/processed'
    type_set = {'company', 'field'}
    emb_df = pd.DataFrame(columns=['label', 'type'])
    vectors = np.empty((0,128))
    for node_type in type_set:
        kv = KeyedVectors.load(osp.join(path, f'{node_type}_metapath2vec'))
        tmp = list(zip(kv.index_to_key, [node_type for i in range(len(kv))]))
        emb_df = emb_df.append(pd.DataFrame(tmp, columns=['label', 'type']), ignore_index=True)
        vectors = np.append(vectors, kv.vectors, axis=0)
        # emb_df['label'].append(pd.Series(kv.index_to_key), ignore_index=True)
        # emb_df['type'].append(pd.Series([node_type for i in range(len(kv))]), ignore_index=True)
        # emb_df['vector'].append(pd.Series(kv.vectors, ignore_index=True)

    emb_df[['label','type']].to_csv(osp.join(path, 'metapath2vec_metadata.tsv'), sep='\t', index=False)
    np.savetxt(osp.join(path, 'metapath2vec_vectors.tsv'), vectors, delimiter='\t')
        