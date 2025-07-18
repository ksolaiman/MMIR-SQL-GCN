import json
import utils
import numpy as np
import os
import set_train_test_data_and_distance_matrix as stt
from datetime import datetime
import pickle


def fx_calc_map_label_detailed(
    db_items, query_items, dist, ground_dist, k,
    threshold, normalize_ap, item2idx,
    save_pr_curve_flag=False, pr_curve_dir=None, label=None
):
    """
    Compute Mean Average Precision (MAP) for retrieval.

    Parameters:
    - db_items: list of item IDs in the database (retrieval set)
    - query_items: list of item IDs for queries
    - dist: predicted distance matrix
    - ground_dist: ground truth distance matrix
    - k: cutoff rank
        > 'k' controls how many of the top-ranked results are evaluated.
        > For example, MAP@5 evaluates only the top 5 retrieved items.
        > Special values:
            * k=0  : means "use all results" (MAP@all)
            * k=-1 : means "evaluate at [50, all]"
    - threshold: distance threshold for relevance (e.g., 2 or 3)
    - normalize_ap: "p_over_r" or "p_over_relAll"
        > Controls the formula for averaging precision.
        > "p_over_r"    : sum of precisions / number of true positives retrieved
        > "p_over_relAll": sum of precisions / number of all relevant items in ground truth
    - item2idx: mapping from item to index

    Returns:
    - List of MAP scores at each k
        > For k = [50, 100]
        > returns [MAP@50, MAP@100]
    """

    numcases = len(db_items)
    query_indices = [item2idx[item] for item in query_items]
    db_indices = [item2idx[item] for item in db_items]

    trunc_dist = dist[np.ix_(query_indices, db_indices)]
    trunc_gt = ground_dist[np.ix_(query_indices, db_indices)]

    # Sort distances to get ranking; Returns the sorted indices
    ord = trunc_dist.argsort(1)

    # Handle special k values
    if k == 0:
        # Special: evaluate on the full ranked list (MAP@all)
        k = [numcases]
    if k == -1:
        # Special: evaluate at two cutoffs
        ks = [50, numcases]
    else:
        ks = k

    def calMAP(_k):
        _res = []
        recall = np.zeros((len(query_indices), len(db_indices)))
        precision = np.zeros((len(query_indices), len(db_indices)))
        non_zero_indices = []

        for i in range(len(query_indices)):
            predicted_order = ord[i]
            # Count of all relevant items in ground truth
            relAll = sum([item < threshold for item in trunc_gt[i]])
            if relAll == 0:
                continue
            non_zero_indices.append(i)

            p = 0.0
            r = 0.0
            total_predicted_relevant = 0

            for j in range(_k):                                     # Iterate over the top-k retrieved (predicted) items for this query (i)
                if trunc_dist[i][predicted_order[j]] < threshold:   # Predicted item's (j) distance for this query (i) is below threshold: model considers this retrieved item relevant
                    total_predicted_relevant += 1
                    if trunc_gt[i][predicted_order[j]] < threshold: # Ground-truth distance for this item (j) for this query (i)  is also below threshold: this is a true positive
                        r += 1
                        p += (r / (j + 1))                          # Accumulate precision at this rank (correct so far / rank position)

                if total_predicted_relevant > 0:
                    precision[i, j] = (r / total_predicted_relevant)
                else:
                    precision[i, j] = 0.0
                recall[i, j] = (r / relAll)

            # After this loop:
            # r      = number of true positives retrieved for this query (correctly predicted relevant items)
            # relAll = total number of relevant items in ground truth for this query

            # Apply chosen normalization scheme
            if r > 0:
                if normalize_ap == "p_over_r":
                    _res.append(p / r)
                else:
                    _res.append(p / relAll)
            else:
                _res.append(0)

        if save_pr_curve_flag and pr_curve_dir and label:
            recall_levels, mean_precision = compute_interpolated_pr_curve(precision, recall)
            save_pr_curve_to_disk(recall_levels, mean_precision, pr_curve_dir, label, 'all' if _k == numcases else _k)

        map_score = np.mean(_res)
        # print(f"\r MAP@{_k}: {map_score:.4f}")
        print(f"\r MAP@{'all' if _k == numcases else _k}: {map_score:.4f}")
        return map_score

    res = []
    for _k in ks:       
        res.append(calMAP(_k))

    # e,g. 
    # for k=[50, 100], returns [MAP@50, MAP@100]; 
    # for k=0, returns [MAP@all]; 
    # for k=-1, returns [MAP@50, MAP@all]
    return res

def compute_interpolated_pr_curve(precision, recall, n_points=11):
    """
    Interpolate precision at standardized recall levels.
    Returns averaged PR curve across queries.
    """
    recall_levels = np.linspace(0, 1, n_points)
    precision_values = []

    for query_no in range(recall.shape[0]):
        precisions_at_levels = []
        for level in recall_levels:
            mask = recall[query_no] >= level
            if np.any(mask):
                precisions_at_levels.append(np.max(precision[query_no][mask]))
            else:
                precisions_at_levels.append(0)
        precision_values.append(precisions_at_levels)

    mean_precision = np.mean(precision_values, axis=0)
    return recall_levels, mean_precision

def save_pr_curve_to_disk(recall_levels, mean_precision, outdir, label, k=None):
    outdir = str(outdir)  # ensure string!
    os.makedirs(outdir, exist_ok=True)
    suffix = f"_k@{k}" if k is not None else ""
    with open(f"{outdir}/{label}{suffix}_recall.pkl", "wb") as f:
        pickle.dump(recall_levels, f)
    with open(f"{outdir}/{label}{suffix}_precision.pkl", "wb") as f:
        pickle.dump(mean_precision, f)

def evaluate_model_MAP(
    model_name,
    all_data,
    test_predicted_dist,
    test_ground_dist,
    test_item2idx,
    K_VALUES,
    THRESHOLD,
    NORMALIZE_AP,
    SAVE_PR,
    PR_DIR
):
    # --------------------------------------------------------
    # RESULTS_SAVING_DIRECTORY
    # --------------------------------------------------------
    timestamp = datetime.now().strftime("%m%d%Y%H%M")
    results_saving_dir = PR_DIR+'/'+model_name+'/'+timestamp              # PR_DIR + model_name + timestamp

    testset = all_data['testset']
    texttestset = all_data['text_testset']
    imagetestset = all_data['image_testset']
    videotestset = all_data['video_testset']

    EVAL_PAIRS = [
        (texttestset, texttestset, "Text->Text"),
        (videotestset, texttestset, "Video->Text"),
        (imagetestset, texttestset, "Image->Text"),
        
        (videotestset, videotestset, "Video->Video"),
        (texttestset, videotestset, "Text->Video"),
        (imagetestset, videotestset, "Image->Video"),
        
        (imagetestset, imagetestset, "Image->Image"),
        (texttestset, imagetestset, "Text->Image"),
        (videotestset, imagetestset, "Video->Image"),

        # (texttestset, texttestset + imagetestset + videotestset,  "Text->Combined"), # gives different result than just passing testset 
        (texttestset, testset,  "Text->All"),
        (videotestset, testset, "Video->All"),
        (imagetestset, testset, "Image->All")
    ]

    # --------------------------------------------------------
    # MAP calculation for each modality
    # --------------------------------------------------------
    maps = []
    for queryset, targetset, label in EVAL_PAIRS:
        print(f"{label}:")
        scores = fx_calc_map_label_detailed(
            db_items=targetset,
            query_items=queryset,
            dist=test_predicted_dist,
            ground_dist=test_ground_dist,
            k=K_VALUES,
            threshold=THRESHOLD,
            normalize_ap=NORMALIZE_AP,
            item2idx=test_item2idx,
            save_pr_curve_flag=SAVE_PR,
            pr_curve_dir=results_saving_dir,
            label=label.replace('->', '_')
        )
        # print(f"{label}: MAP = {scores}")
        maps.append(scores)

    # print(maps)
    mean_maps = [sum(col) / len(col) for col in zip(*maps)]

    ks = (['50', 'all'] if K_VALUES == -1 else 
      ['all'] if K_VALUES == 0 else 
      CONFIG.get('k_values', 0))
    
    print(f"\n Mean MAPs @ {ks} = {mean_maps}")

    # --------------------------------------------------------
    # SAVE MAP calculation result
    # --------------------------------------------------------
    results_lines = []
    for kix, k in enumerate(ks):
        results_lines.append(f"#### MAP@{k} Results ####\n")
        for qix, (queryset, targetset, label) in enumerate(EVAL_PAIRS):
            results_lines.append(f"{label}: MAP = {maps[qix][kix]:.4f}")
        results_lines.append(f"Mean MAP = {mean_maps[kix]:.4f}\n")

    os.makedirs(results_saving_dir, exist_ok=True)
    with open(os.path.join(results_saving_dir, 'MAP_results.txt'), 'w') as f:
        for line in results_lines:
            f.write(line + '\n')

    # --------------------------------------------------------
    # MAP calculation completed
    # --------------------------------------------------------


if __name__ == "__main__":
    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------
    CONFIG = utils.load_config()
    DATA_SOURCE = CONFIG.get('database_source', 'local')
    THRESHOLD = CONFIG.get('relevance_threshold', 3)
    NORMALIZE_AP = CONFIG.get('normalize_ap', 'p_over_r')
    SAVE_PR = CONFIG.get('save_pr', False)                  # or just say, true or false, easy change
    PR_DIR = CONFIG.get('pr_curve_dir', 'prcurve')
    DATASET_DIR = CONFIG.get('dataset_dir', "dataset")
    K_VALUES = CONFIG.get('k_values', 0)

    # --------------------------------------------------------
    # LOAD test splits from dataset folder
    # Note: we don't re-access DB or re-prepare splits anymore.
    # These IDs were generated once and saved under DATASET_DIR.
    # --------------------------------------------------------
    all_data = stt.load_all_splits(DATASET_DIR)

    testset = all_data['testset']
    texttestset = all_data['text_testset']
    imagetestset = all_data['image_testset']
    videotestset = all_data['video_testset']

    print("\n Loaded test split IDs from disk:")
    print(f"- Combined testset size: {len(testset)}")
    print(f"- Text testset: {len(texttestset)}")
    print(f"- Image testset: {len(imagetestset)}")
    print(f"- Video testset: {len(videotestset)}")

    # --------------------------------------------------------
    # LOAD pre-saved distance matrices and mappings for testset
    # These were calculated and saved separately already.
    # test_ground_dist is same for all models
    # --------------------------------------------------------
    test_ground_dist, EARS_test_predicted_dist, test_item2idx, test_idx2item = stt.load_dist_matrices(
        "test", base_dir=DATASET_DIR
    )

    print("\n Loaded test distance matrices from disk:")
    print(f"- Ground truth matrix shape: {test_ground_dist.shape}")
    print(f"- Predicted matrix shape: {EARS_test_predicted_dist.shape}")


    evaluate_model_MAP('EARS', all_data, EARS_test_predicted_dist,
                       test_ground_dist, test_item2idx,
                       K_VALUES, THRESHOLD, NORMALIZE_AP, SAVE_PR, PR_DIR)

    