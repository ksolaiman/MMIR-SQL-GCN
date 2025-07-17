import pickle
import os
import utils
from collections import defaultdict
from set_train_test_data_and_distance_matrix import train_validation_split
import random
import numpy as np

POSITIVE_THRESHOLD = 3
NEGATIVE_THRESHOLD = 6
TARGET_POS_COUNT_PER_MODALITY = 1000
TARGET_NEG_COUNT_PER_MODALITY = 1000

# Config
config = utils.load_config()
DATASET_DIR = config.get("dataset_dir")

# Create item2modality mapping
def build_item2modality(image_list, video_list, text_list):
    mapping = {}
    for item in image_list:
        mapping[item] = "image"
    for item in video_list:
        mapping[item] = "video"
    for item in text_list:
        mapping[item] = "text"
    return mapping

def create_positive_and_negative_pairs_from_set_of_items(train_pool, item2idx, item2modality, save_pairs=False):

    all_positive_pairs = list()
    all_negative_pairs = list()

    for qid in train_pool:
        q_idx = item2idx[qid]
        mod_q = item2modality[qid]
        
        for cid in train_pool:
            if qid == cid:
                continue
            c_idx = item2idx[cid]
            mod_c = item2modality[cid]

            ced = distance_matrix[q_idx][c_idx]
            pair_modality = (mod_q, mod_c)

            if ced <= POSITIVE_THRESHOLD:
                # positive_pairs_by_modality[pair_modality].append((qid, cid, ced))
                all_positive_pairs.append((qid, cid, ced, pair_modality))
            elif POSITIVE_THRESHOLD < ced <= NEGATIVE_THRESHOLD:
                # negative_pairs_by_modality[pair_modality].append((qid, cid, ced))
                all_negative_pairs.append((qid, cid, ced, pair_modality))

    if save_pairs:
        # ✅ Save pair buckets for future sampling
        pair_bucket_dir = os.path.join(DATASET_DIR, "femmir_pair_lists")
        os.makedirs(pair_bucket_dir, exist_ok=True)
        
        with open(os.path.join(pair_bucket_dir, "all_positive_pairs.pkl"), "wb") as f:
            pickle.dump(all_positive_pairs, f)

        with open(os.path.join(pair_bucket_dir, "all_negative_pairs.pkl"), "wb") as f:
            pickle.dump(all_negative_pairs, f)


def create_positive_and_negative_pairs_from_set_of_items_vectorized(train_pool, item2idx, item2modality, save_pairs=False):

    all_positive_pairs = list()
    all_negative_pairs = list()
    train_pool_array = np.array(train_pool)
    train_pool_indices = np.array([item2idx[qid] for qid in train_pool])
    train_pool_modalities = np.array([item2modality[qid] for qid in train_pool])

    for i, qid in enumerate(train_pool):
        q_idx = item2idx[qid]
        mod_q = item2modality[qid]

        # Get candidate indices excluding self
        mask_not_self = train_pool_array != qid
        cids = train_pool_array[mask_not_self]
        c_indices = train_pool_indices[mask_not_self]
        c_modalities = train_pool_modalities[mask_not_self]

        ceds = distance_matrix[q_idx, c_indices]

        pos_mask = ceds <= POSITIVE_THRESHOLD
        neg_mask = (ceds > POSITIVE_THRESHOLD) & (ceds <= NEGATIVE_THRESHOLD)

        # Append positives
        all_positive_pairs.extend([
            (qid, cid, float(ced), (mod_q, mod_c))
            for cid, ced, mod_c in zip(cids[pos_mask], ceds[pos_mask], c_modalities[pos_mask])
        ])

        # Append negatives
        all_negative_pairs.extend([
            (qid, cid, float(ced), (mod_q, mod_c))
            for cid, ced, mod_c in zip(cids[neg_mask], ceds[neg_mask], c_modalities[neg_mask])
        ])

    
    if save_pairs:
        # ✅ Save pair buckets for future sampling
        pair_bucket_dir = os.path.join(DATASET_DIR, "femmir_pair_lists")
        os.makedirs(pair_bucket_dir, exist_ok=True)
        
        with open(os.path.join(pair_bucket_dir, "all_positive_pairs.pkl"), "wb") as f:
            pickle.dump(all_positive_pairs, f)

        with open(os.path.join(pair_bucket_dir, "all_negative_pairs.pkl"), "wb") as f:
            pickle.dump(all_negative_pairs, f)


def load_pool(subpath):
    full_path = os.path.join(DATASET_DIR, subpath)
    with open(full_path, "rb") as f:
        return pickle.load(f)

image_pool = load_pool("image/image_trainpool.pkl")
video_pool = load_pool("video/video_trainpool.pkl")
text_pool = load_pool("text/text_trainpool.pkl")

# image_train, image_val = train_validation_split(image_pool)
# video_train, video_val = train_validation_split(video_pool)
# text_train, text_val = train_validation_split(text_pool)

# print(f"Image train: {len(image_train)}, val: {len(image_val)}")
# print(f"Video train: {len(video_train)}, val: {len(video_val)}")
# print(f"Text train: {len(text_train)}, val: {len(text_val)}")

# train_items = image_train + video_train + text_train
# val_items = image_val + video_val + text_val


# def create_femmir_pairs(mode='train'):
train_pool = load_pool("trainpool.pkl")
item2modality = build_item2modality(image_pool, video_pool, text_pool)

# Load distance matrix and mappings
with open(os.path.join(DATASET_DIR, "dist_matrices/train/ground_distance_matrix.pkl"), "rb") as f:
    distance_matrix = pickle.load(f)

with open(os.path.join(DATASET_DIR, "dist_matrices/train/item2idx.pkl"), "rb") as f:
    item2idx = pickle.load(f)


create_positive_and_negative_pairs_from_set_of_items_vectorized(train_pool[0:1000], item2idx, item2modality, True)


# # Keep modality-wise buckets
# # keep track of positive/negative pairs by modality pair type (e.g. ("text","image"), ("video","video"), etc.).
# positive_pairs_by_modality = defaultdict(list)
# negative_pairs_by_modality = defaultdict(list)



# print(positive_pairs_by_modality.keys())  
# # print(positive_pairs_by_modality.values())  

# # ✅ Save pair buckets for future sampling
# pair_bucket_dir = os.path.join(DATASET_DIR, "femmir_pair_lists")
# os.makedirs(pair_bucket_dir, exist_ok=True)

# with open(os.path.join(pair_bucket_dir, "positive_pairs_by_modality.pkl"), "wb") as f:
#     pickle.dump(positive_pairs_by_modality, f)

# with open(os.path.join(pair_bucket_dir, "negative_pairs_by_modality.pkl"), "wb") as f:
#     pickle.dump(negative_pairs_by_modality, f)



# final_train_pairs = []

# for modality_type in positive_pairs_by_modality:
#     pos_pool = positive_pairs_by_modality[modality_type]
#     neg_pool = negative_pairs_by_modality.get(modality_type, [])

#     sampled_pos = random.sample(pos_pool, min(TARGET_POS_COUNT_PER_MODALITY, len(pos_pool)))
#     sampled_neg = random.sample(neg_pool, min(TARGET_NEG_COUNT_PER_MODALITY, len(neg_pool)))

#     final_train_pairs.extend(sampled_pos + sampled_neg)
