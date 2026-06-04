import pickle
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from metric_utils import *
import sklearn
from sklearn import covariance
from metric_utils import get_measures


parser = argparse.ArgumentParser(description='Evaluates an OOD Detector',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', default=1., type=str)
parser.add_argument('--use_trained_params', default=0, type=int)
parser.add_argument('--pro_length', default=16, type=int)
parser.add_argument('--dir', default='output', type=str)
parser.add_argument('--score_thr', default=0.1, type=float)
parser.add_argument('--max_in_samples', default=200000, type=int,
                    help='Cap ID eval samples for speed; set <=0 to use all')
parser.add_argument('--max_out_samples', default=50000, type=int,
                    help='Cap OOD eval samples for speed; set <=0 to use all')
parser.add_argument('--search_batch_size', default=100000, type=int,
                    help='FAISS query batch size')
args = parser.parse_args()

assert args.use_trained_params == 0

name = args.name
length = 256 # in the oag-dino, the output bbox_embd and the enhanced_text_features are both 256-dimensional features

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


# ID data

normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))


def cap_samples(x, cap, name):
    if cap is None or cap <= 0 or x.shape[0] <= cap:
        return x
    print(f"[speed] truncate {name}: {x.shape[0]} -> {cap}")
    return x[:cap]


def knn_last_distance(index, queries, k, batch_size, name):
    n = queries.shape[0]
    out = np.empty((n,), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        D, _ = index.search(queries[s:e], k)
        out[s:e] = D[:, -1]
        print(f"[faiss] {name} k={k}: {e}/{n}")
    return out


id_train_data = np.load('DDERT_OUTPUTS/bdd_coco_2tasks/id-pen_maha_train-79.npy')  
id_train_data_logits = np.load('DDERT_OUTPUTS/bdd_coco_2tasks/id-logits_maha_train-79.npy')  
print(id_train_data.shape)
id_train_data = torch.from_numpy(id_train_data).reshape(-1, id_train_data.shape[-1]) # [:, :-1]
id_train_data = id_train_data[:, :length]


print(id_train_data.shape)
print(id_train_data.shape)
print(id_train_data.shape)


all_logits_in = np.load('DDERT_OUTPUTS/bdd_coco_2tasks/id-logits-79.npy')
all_data_in = np.load('DDERT_OUTPUTS/bdd_coco_2tasks/id-pen-79.npy')
print(all_data_in.shape)
all_data_in = torch.from_numpy(all_data_in).reshape(-1, all_data_in.shape[-1])
all_logits_out = np.load('DDERT_OUTPUTS/bdd_coco_2tasks/ood-logits-79.npy')
all_data_out = np.load('DDERT_OUTPUTS/bdd_coco_2tasks/ood-pen-79.npy')
print(all_data_out.shape)
all_data_out = torch.from_numpy(all_data_out).reshape(-1, all_data_out.shape[-1])


print(all_data_in.shape)
print(all_data_out.shape)

all_logits_in = torch.from_numpy(all_logits_in)
all_logits_out = torch.from_numpy(all_logits_out)
id_train_data_logits = torch.from_numpy(id_train_data_logits)

print(all_logits_in.shape)
print(all_logits_out.shape)


all_data_in = all_data_in[torch.where(all_logits_in.max(dim=1)[0] > args.score_thr)].numpy()
all_data_out = all_data_out[torch.where(all_logits_out.max(dim=1)[0] > args.score_thr)].numpy()

all_data_in = cap_samples(all_data_in, args.max_in_samples, 'all_data_in')
all_data_out = cap_samples(all_data_out, args.max_out_samples, 'all_data_out')



print(all_data_in.shape)
print(all_data_out.shape)

id = 0
T = 1
scores_in = []
scores_ood_test = []

mean_list = []
covariance_list = []


# knn score
id_train_data = prepos_feat(id_train_data)
all_data_in = prepos_feat(all_data_in)
all_data_out = prepos_feat(all_data_out)


import faiss
index = faiss.IndexFlatL2(id_train_data.shape[1])
index.add(id_train_data)
index.add(id_train_data)
for K in [1, 5, 10, 25, 50, 100, 200, 300, 400, 500]:
    print(f"\n[run] start K={K}")
    scores_in = -knn_last_distance(index, all_data_in, K, args.search_batch_size, 'ID')
    all_results = []
    all_score_ood = []
    # for ood_dataset, food in food_all.items():
    scores_ood_test = -knn_last_distance(index, all_data_out, K, args.search_batch_size, 'OOD')
    all_score_ood.extend(scores_ood_test)

    print('-----------------id------------------')
    print(np.percentile(scores_in, 5))
    print(np.percentile(scores_in, 25))
    print(np.percentile(scores_in, 50))
    print(np.percentile(scores_in, 75))
    print(np.percentile(scores_in, 95))
    print('-----------------ood------------------')
    print(np.percentile(scores_ood_test, 5))
    print(np.percentile(scores_ood_test, 25))
    print(np.percentile(scores_ood_test, 50))
    print(np.percentile(scores_ood_test, 75))
    print(np.percentile(scores_ood_test, 95))

    results = get_measures(scores_in, scores_ood_test, plot=False)


    print_measures(results[0], results[1], results[2], f'KNN k={K}')
    print()


