import numpy as np
import pandas as pd
from google_landmark_retrieval.constants import IMG_SIZE, RETRIEVAL_DATA_PATH
# from sklearn.neighbors import KDTree
from google_landmark_retrieval.data import *
import faiss

test_embeddings = np.load('/home/lyan/Documents/kaggle/google_landmark_retrieval/retrieval_sample_submission.csv.embeddings.npy')
index_embeddings = np.load('/home/lyan/Documents/kaggle/google_landmark_retrieval/index.csv.embeddings.npy')


submit_ds = LandmarkDataset('%s' % RETRIEVAL_DATA_PATH,
                     '/var/ssd_1t/google_landmark_retrieval_test/', 'retrieval_sample_submission.csv', None, test=True)


index_ds = LandmarkDataset('%s' % RETRIEVAL_DATA_PATH,
                     '/var/ssd_1t/google_landmark_retrieval_index/', 'index.csv', None, test=True)

index = faiss.IndexFlatL2(1024)
index.add(index_embeddings)
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)
print('index on gpu')


data = submit_ds.data

for i in tqdm(range(test_embeddings.shape[0])):
    ids = index.search(test_embeddings[i].reshape(1, -1), k=75)
    dist, ids = ids
    ids = ids[0]
    idx = submit_ds.data.iloc[i, 0]

    result_ids = []
    for im in ids:
        result_ids.append(index_ds.data.iloc[im, 0])

    result_ids = ' '.join(result_ids)
    data.iloc[i, 1] = result_ids

submit_df = pd.read_csv('/home/lyan/Documents/kaggle_data/google_landmark_retrieval/retrieval_sample_submission.csv')
data = pd.merge(data, submit_df, how='outer', on=['id'])
data = data.drop('images_y',axis=1)
data.columns = ['id','images']
data.to_csv('submit.csv',index=False)