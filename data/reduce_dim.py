import pandas as pd
import torch
import numpy as np
from sklearn.decomposition import PCA

def PCA(cls_embedding, target_dim, dataname):
    last_hidden_states_np = cls_embedding

    pca = PCA(n_components=target_dim)
    reduced_embeddings_np = pca.fit_transform(last_hidden_states_np.reshape(-1, last_hidden_states_np.shape[-1]))

    reduced_embeddings = torch.tensor(reduced_embeddings_np)

    print(reduced_embeddings.shape)
    np.save(f'./{dataname}/item_features_{target_dim}.npy', reduced_embeddings)


if __name__ == '__main__':
    dataname = "KU"
    cls_embedding = np.load(f'./{dataname}/item_features_768.npy')
    print(cls_embedding.shape)
    target_dim = 64
    PCA(cls_embedding, target_dim, dataname)
