import pickle
import os
import utils
from collections import defaultdict
from set_train_test_data_and_distance_matrix import train_validation_split, stratified_train_validation_split_by_modality
import random
import numpy as np
from joblib import Parallel, delayed

POSITIVE_THRESHOLD = 3
NEGATIVE_THRESHOLD = 6
TARGET_POS_COUNT_PER_MODALITY = 1000
TARGET_NEG_COUNT_PER_MODALITY = 1000

# Config
config = utils.load_config()
DATASET_DIR = config.get("dataset_dir")

# Load distance matrix and mappings
with open(os.path.join(DATASET_DIR, "dist_matrices/train/ground_distance_matrix.pkl"), "rb") as f:
    distance_matrix = pickle.load(f)

with open(os.path.join(DATASET_DIR, "dist_matrices/train/item2idx.pkl"), "rb") as f:
    item2idx = pickle.load(f)

with open(os.path.join(DATASET_DIR, "dist_matrices/train/item2modality.pkl"), "rb") as f:
    item2modality = pickle.load(f)


###### Currently Unused ######
def process_qid(qid, train_pool, item2idx, item2modality, distance_matrix):
    '''
    returns all positive and negative (query, target) pairs for the object-item "qid"
    '''
    q_idx = item2idx[qid]
    mod_q = item2modality[qid]

    # Initialize local storage for this qid
    positive_pairs = []
    negative_pairs = []

    # Use masking logic or nested loop here
    for cid in train_pool:
        if qid == cid:
            continue
        c_idx = item2idx[cid]
        mod_c = item2modality[cid]
        ced = distance_matrix[q_idx][c_idx]
        pair_modality = (mod_q, mod_c)

        if ced <= POSITIVE_THRESHOLD:
            positive_pairs.append((qid, cid, float(ced), pair_modality))
        elif POSITIVE_THRESHOLD < ced <= NEGATIVE_THRESHOLD:
            negative_pairs.append((qid, cid, float(ced), pair_modality))

    return positive_pairs, negative_pairs

def process_qid_vectorized(qid, NOS_mod,
                           train_pool_array, train_pool_indices, train_pool_modalities, 
                           item2idx, item2modality, distance_matrix, RANDOM_SEED=42):
    '''
    returns "NOS_mod" number of positive and negative (query, target) pairs per modality for the object-item "qid"
    '''
    
    # random.seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED + hash(qid) % (2**32))
    q_idx = item2idx[qid]
    mod_q = item2modality[qid]

    # Exclude self
    mask_not_self = train_pool_array != qid
    cids = train_pool_array[mask_not_self]
    c_indices = train_pool_indices[mask_not_self]
    c_modalities = train_pool_modalities[mask_not_self]

    ceds = distance_matrix[q_idx, c_indices]

    pos_mask = ceds <= POSITIVE_THRESHOLD
    neg_mask = (ceds > POSITIVE_THRESHOLD) & (ceds <= NEGATIVE_THRESHOLD)

    # positive_pairs = [
    #     (qid, cid, float(ced), (mod_q, mod_c))
    #     for cid, ced, mod_c in zip(cids[pos_mask], ceds[pos_mask], c_modalities[pos_mask])
    # ]

    # negative_pairs = [
    #     (qid, cid, float(ced), (mod_q, mod_c))
    #     for cid, ced, mod_c in zip(cids[neg_mask], ceds[neg_mask], c_modalities[neg_mask])
    # ]


    # Create modality-based buckets
    pos_by_mod = defaultdict(list)
    neg_by_mod = defaultdict(list)

    # Group positive pairs by candidate modality
    for cid, ced, mod_c in zip(cids[pos_mask], ceds[pos_mask], c_modalities[pos_mask]):
        pos_by_mod[mod_c].append((qid, cid, float(ced), (mod_q, mod_c)))

    # Group negative pairs by candidate modality
    for cid, ced, mod_c in zip(cids[neg_mask], ceds[neg_mask], c_modalities[neg_mask]):
        neg_by_mod[mod_c].append((qid, cid, float(ced), (mod_q, mod_c)))

    # Stratified sampling: 5 per modality (if available)
    modalities = ["text", "image", "video"]

    positive_pairs = []
    negative_pairs = []

    for m in modalities:
        positive_pairs.extend(rng.sample(pos_by_mod[m], min(NOS_mod, len(pos_by_mod[m]))))
        negative_pairs.extend(rng.sample(neg_by_mod[m], min(NOS_mod, len(neg_by_mod[m]))))

    return qid, positive_pairs, negative_pairs

###### Currently Unused ######
def chunkwise_cpnpsi(train_pool, NOS_mod, item2idx, item2modality, distance_matrix, save_pairs=False, RANDOM_SEED=42):
    '''
    Create "NOS_mod" number of positive and negative pairs for every item in "train_pool"
    and
    Save them all in one file (if NOS_mod is larger, it is easier to save with this function in multiple chunks)
    and uses Parallel processing using joblib and vectorize the inner for loop
    '''
    train_pool_array = np.array(train_pool)
    train_pool_indices = np.array([item2idx[qid] for qid in train_pool])
    train_pool_modalities = np.array([item2modality[qid] for qid in train_pool])

    start_idx = 0 
    increment = 1000
    num_chunks = int(np.ceil(len(train_pool) / increment))

    for i in range(0, num_chunks):
        end_idx = start_idx + increment
        end_idx = min(end_idx, len(train_pool))  # ensure not out of bounds
    
        results = Parallel(n_jobs=8, backend="threading")(delayed(process_qid_vectorized)(
                qid, NOS_mod,
                train_pool_array,
                train_pool_indices,
                train_pool_modalities,
                item2idx, 
                item2modality, distance_matrix,
                RANDOM_SEED
                ) for qid in train_pool[start_idx:end_idx])
        
        start_idx += increment

        # Flatten
        # all_positive_pairs = [item for result in results for item in result[0]]
        # all_negative_pairs = [item for result in results for item in result[1]]

        # Flatten into dictionaries keyed by qid
        all_positive_pairs = {qid: positive for qid, positive, _ in results}
        all_negative_pairs = {qid: negative for qid, _, negative in results}

        if save_pairs:
            # Save pair buckets for future sampling
            pair_bucket_dir = os.path.join(DATASET_DIR, "femmir_pair_lists_v2")
            os.makedirs(pair_bucket_dir, exist_ok=True)
            
            with open(os.path.join(pair_bucket_dir, "all_positive_pairs"+ str(i) +".pkl"), "wb") as f:
                pickle.dump(all_positive_pairs, f)

            with open(os.path.join(pair_bucket_dir, "all_negative_pairs"+ str(i) +".pkl"), "wb") as f:
                pickle.dump(all_negative_pairs, f)

        print(f"Done chunk {i+1}/{num_chunks}: {end_idx} items processed")

def cpnpsi(train_pool, NOS_mod, item2idx, item2modality, distance_matrix, save_pairs=False, RANDOM_SEED=42):
    '''
    Create "NOS_mod" number of positive and negative pairs for every item in "train_pool"
    and Save them all in one file (if NOS_mod is small, it is faster to save and can be done in this function)
    and uses Parallel processing using joblib and vectorize the inner for loop
    '''
    train_pool_array = np.array(train_pool)
    train_pool_indices = np.array([item2idx[qid] for qid in train_pool])
    train_pool_modalities = np.array([item2modality[qid] for qid in train_pool])

    results = Parallel(n_jobs=8, backend="threading")(delayed(process_qid_vectorized)(
                qid, NOS_mod,
                train_pool_array,
                train_pool_indices,
                train_pool_modalities,
                item2idx, 
                item2modality, distance_matrix,
                RANDOM_SEED
                ) for qid in train_pool)
     
    # Flatten
    # all_positive_pairs = [item for result in results for item in result[0]]
    # all_negative_pairs = [item for result in results for item in result[1]]

    # Flatten into dictionaries keyed by qid
    # Would make it easier to just create pairs from an object id
    all_positive_pairs = {qid: positive for qid, positive, _ in results}
    all_negative_pairs = {qid: negative for qid, _, negative in results}

    if save_pairs:
        # Save pair buckets for future sampling
        pair_bucket_dir = os.path.join(DATASET_DIR, config.get("pair_save_dir"))
        os.makedirs(pair_bucket_dir, exist_ok=True)
        
        with open(os.path.join(pair_bucket_dir, "positive_pairs_by_queryid.pkl"), "wb") as f:
            pickle.dump(all_positive_pairs, f)

        with open(os.path.join(pair_bucket_dir, "negative_pairs_by_queryid.pkl"), "wb") as f:
            pickle.dump(all_negative_pairs, f)


###### Currently Unused ######
def create_positive_and_negative_pairs_from_set_of_items_all_vectorized(train_pool, item2idx, item2modality, distance_matrix, save_pairs=False):
    '''
    Create all positive and negative pairs for every item in "train_pool"
    and Save them all in one file (if NOS_mod is small, it is faster to save and can be done in this function)
    and vectorize the both for loop
    '''
    N = len(train_pool)
    idx_array = np.array([item2idx[qid] for qid in train_pool])
    modalities = np.array([item2modality[qid] for qid in train_pool])

    # Create full meshgrid of query/candidate indices
    q_idx_grid, c_idx_grid = np.meshgrid(idx_array, idx_array, indexing='ij')
    q_mods, c_mods = np.meshgrid(modalities, modalities, indexing='ij')

    # Mask self-pairs
    self_mask = q_idx_grid != c_idx_grid

    # Extract CED values
    ced_matrix = distance_matrix[q_idx_grid, c_idx_grid]

    # Apply masks
    pos_mask = (ced_matrix <= POSITIVE_THRESHOLD) & self_mask
    neg_mask = ((ced_matrix > POSITIVE_THRESHOLD) & (ced_matrix <= NEGATIVE_THRESHOLD)) & self_mask

    # Extract positive/negative qid-cid pairs
    qids = np.array(train_pool)
    qid_grid, cid_grid = np.meshgrid(qids, qids, indexing='ij')

    all_positive_pairs = list(zip(qid_grid[pos_mask], cid_grid[pos_mask], ced_matrix[pos_mask],
                            zip(q_mods[pos_mask], c_mods[pos_mask])))

    all_negative_pairs = list(zip(qid_grid[neg_mask], cid_grid[neg_mask], ced_matrix[neg_mask],
                            zip(q_mods[neg_mask], c_mods[neg_mask])))


    if save_pairs:
        # ✅ Save pair buckets for future sampling
        pair_bucket_dir = os.path.join(DATASET_DIR, "femmir_pair_lists")
        os.makedirs(pair_bucket_dir, exist_ok=True)
        
        with open(os.path.join(pair_bucket_dir, "all_positive_pairs.pkl"), "wb") as f:
            pickle.dump(all_positive_pairs, f)

        with open(os.path.join(pair_bucket_dir, "all_negative_pairs.pkl"), "wb") as f:
            pickle.dump(all_negative_pairs, f)


###### Currently Unused ######
def create_positive_and_negative_pairs_from_set_of_items_vectorized(train_pool, item2idx, item2modality, distance_matrix, save_pairs=False):
    '''
    Create all positive and negative pairs for every item in "train_pool"
    and Save them all in one file (if NOS_mod is small, it is faster to save and can be done in this function)
    and vectorize the inner for loop
    '''
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



def create_pairs_for_full_training_pool(NOS_mod=5, save_pairs=True, RANDOM_SEED=42):   
    '''
    Creating certain number of modality-stratified positive and negative (query, target) pairs for all of training pool.
    Saved pairs could be access by query-id of the object/item.
    IDEA is in future, whatever split I choose - KFold/80-20/balanced/unbalanced split,
        just access the pairs of the id's there and create training data for SIMGNN
    '''
    train_pool = utils.load_pool("trainpool.pkl")
    cpnpsi(train_pool, NOS_mod, item2idx, item2modality, distance_matrix, save_pairs, RANDOM_SEED) 
    
if __name__ == "__main__":
    # create_pairs_for_full_training_pool(5, True, 42)            # already ran and created

    # loading the pairs
    pair_dir = os.path.join(DATASET_DIR, config.get("pair_save_dir"))
    positive_pairs = utils.load_pickle(os.path.join(pair_dir, f"positive_pairs_by_queryid.pkl"))
    negative_pairs = utils.load_pickle(os.path.join(pair_dir, f"negative_pairs_by_queryid.pkl"))

    ObjectID = 466
    print(f"✅ Loaded {len(positive_pairs[ObjectID])} positive pairs and {len(positive_pairs[ObjectID])} negative pairs for {ObjectID}")
    print(f"✅ Total pairs generated: {len(positive_pairs) * (2 * len(positive_pairs[ObjectID]) )}")



####### Reading the split files of positive and negative pairs
'''     # using Parallel
all_files = [os.path.join(pair_dir, f"all_positive_pairs{i}.pkl") for i in range(7)]
all_batches = Parallel(n_jobs=4, backend="threading")(delayed(load_pickle)(f) for f in all_files)
positive_pairs = [item for batch in all_batches for item in batch]

print(f"✅ Loaded {len(positive_pairs)} positive pairs")
'''

'''     # not using Parallel
# Load all positive pairs
for i in range(1):  # 🔁 Adjust range if you have more/less chunks
    fname = os.path.join(pair_dir, f"all_positive_pairs{i}.pkl")
    with open(fname, "rb") as f:
        positive_pairs.extend(pickle.load(f))

print(f"✅ Loaded {len(positive_pairs)} positive pairs")

# Load all negative pairs
for i in range(7):
    fname = os.path.join(pair_dir, f"all_negative_pairs{i}.pkl")
    with open(fname, "rb") as f:
        negative_pairs.extend(pickle.load(f))

print(f"✅ Loaded {len(negative_pairs)} negative pairs")
'''