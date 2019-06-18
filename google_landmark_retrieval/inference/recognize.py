import numpy as np
import pandas as pd
from google_landmark_retrieval.constants import IMG_SIZE, RETRIEVAL_DATA_PATH
# from sklearn.neighbors import KDTree
from google_landmark_retrieval.data import *
import faiss

test_embeddings = np.load('/home/lyan/Documents/kaggle/google_landmark_retrieval/retrieval_sample_submission.csv.embeddings.npy')
train_embeddings = np.load('/home/lyan/Documents/kaggle/google_landmark_retrieval/train.csv.embeddings.npy')


submit_ds = LandmarkDataset('%s' % RETRIEVAL_DATA_PATH,
                     '/var/ssd_1t/google_landmark_retrieval_test/', 'retrieval_sample_submission.csv', None, test=True)


train_ds = LandmarkDataset('%s' % RETRIEVAL_DATA_PATH,
                     '/var/ssd_1t/google_landmark_retrieval_train/', 'train.csv', None, test=True)

index = faiss.IndexFlatL2(512)
index.add(train_embeddings)
res = faiss.StandardGpuResources()
index = faiss.index_cpu_to_gpu(res, 0, index)
print('index on gpu')


data = submit_ds.data

for i in tqdm(range(test_embeddings.shape[0])):
    index_search_result = index.search(test_embeddings[i].reshape(1, -1), k=1)
    dist, image_ids = index_search_result
    data.iloc[i, 1] = ' '.join([train_ds.data.iloc[image_ids[0][0], 2], '0.1'])

data.columns = ['id','landmarks']
submit_df = pd.read_csv('/home/lyan/Documents/kaggle_data/google_landmark_recognition/recognition_sample_submission.csv')
data = pd.merge(data, submit_df, how='outer', on=['id'])
data = data.drop('landmarks_y',axis=1)
data.columns = ['id','landmarks']
data.to_csv('recognition_submit.csv',index=False)
print('done')