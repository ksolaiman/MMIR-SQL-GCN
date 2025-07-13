import json
import pickle
import numpy as np
import os
import set_train_test_data_and_distance_matrix_from_mmir_ground as stt
import calc_MAP_mac_1 as cmac1

# Load config
with open('config.json', 'r') as f:
    CONFIG = json.load(f)

DATA_SOURCE = CONFIG.get('database_source', 'local')        # second param is default
THRESHOLD = CONFIG.get('relevance_threshold', 3)
NORMALIZE_AP = CONFIG.get('normalize_ap', 'p_over_r')

print(f"Config loaded: {CONFIG}")


def load_config(config_path='config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


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
    """

    numcases = len(db_items)
    query_indices = [item2idx[item] for item in query_items]
    db_indices = [item2idx[item] for item in db_items]

    trunc_dist = dist[np.ix_(query_indices, db_indices)]
    trunc_gt = ground_dist[np.ix_(query_indices, db_indices)]

    # Sort distances to get ranking
    ord = trunc_dist.argsort(1)

    # Handle special k values
    if k == 0:
        # Special: evaluate on the full ranked list (MAP@all)
        k = numcases
    if k == -1:
        # Special: evaluate at two cutoffs
        ks = [50, numcases]
    else:
        ks = [k]

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

            for j in range(_k):
                if trunc_dist[i][predicted_order[j]] < threshold:
                    total_predicted_relevant += 1
                    if trunc_gt[i][predicted_order[j]] < threshold:
                        r += 1
                        p += (r / (j + 1))

                if total_predicted_relevant > 0:
                    precision[i, j] = (r / total_predicted_relevant)
                else:
                    precision[i, j] = 0.0
                recall[i, j] = (r / relAll)

            # Apply chosen normalization scheme
            if r > 0:
                if normalize_ap == "p_over_r":
                    _res.append(p / r)
                else:
                    _res.append(p / relAll)
            else:
                _res.append(0)

        map_score = np.mean(_res)
        print(f"MAP@{_k}: {map_score:.4f}")
        return map_score

    res = []
    for _k in ks:
        res.append(calMAP(_k))

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

def save_pr_curve_to_disk(recall_levels, mean_precision, outdir, label):
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/{label}_recall.pkl", "wb") as f:
        pickle.dump(recall_levels, f)
    with open(f"{outdir}/{label}_precision.pkl", "wb") as f:
        pickle.dump(mean_precision, f)


if __name__ == "__main__":
    CONFIG = load_config()
    DATA_SOURCE = CONFIG.get('database_source', 'local')
    THRESHOLD = CONFIG.get('relevance_threshold', 3)
    NORMALIZE_AP = CONFIG.get('normalize_ap', 'p_over_r')
    SAVE_PR = CONFIG.get('save_pr', False)
    PR_DIR = CONFIG.get('pr_curve_dir', 'prcurve')
    # dir = CONFIG.get('dataset_dir', None)

    np.random.seed(42)          # this random seed has no affect in MAP calculation below
                                # but if you pass a seed to Database, it has an effect
    rows = stt.prepare_dataset()
    testset = [row[2] for row in rows if row[2]]
    texttestset = [row[5] for row in rows if row[5]]
    imagetestset = [row[3] for row in rows if row[3]]
    videotestset = [row[4] for row in rows if row[4]]
    testset = sorted(set(texttestset + imagetestset + videotestset))
    ground_dist, predicted_dist, item2idx, idx2item = stt.calc_distance_local_DB(testset)

    # os.makedirs(dir, exist_ok=True)
    # os.chdir(dir)
    # stt.write_dataset(testset, item2idx, idx2item, 'testset')
    # stt.write_dataset(texttestset, item2idx, idx2item, 'texttestset')
    # stt.write_dataset(imagetestset, item2idx, idx2item, 'imagetestset')
    # stt.write_dataset(videotestset, item2idx, idx2item, 'videotestset')
    # stt.save_dist_matrices("test_distance_matrix", ground_dist, predicted_dist)


    # Load matrices and mappings for testdata
    # testset, item2idx, idx2item, ground_dist, predicted_dist = stt.read_dataset(dir+'testset', 
    #                                                                   dir+'test_distance_matrix', 
    #                                                                   dir+'predicted_test_distance_matrix')

    print("Loaded predicted distances:", predicted_dist.shape)
    print("Loaded ground truth distances:", ground_dist.shape)

    # # Prepare dataset splits
    # # direct file reading, no more dataset access, or use above 4 lines
    # with open(dir+'texttestset'+".pkl", "rb") as f:
    #     texttestset  = pickle.load(f)
    # with open(dir+'imagetestset'+".pkl", "rb") as f:
    #     imagetestset  = pickle.load(f)
    # with open(dir+'videotestset'+".pkl", "rb") as f:
    #     videotestset  = pickle.load(f)
        
    print("Test set sizes:")
    print(f"- Text: {len(texttestset)}")
    print(f"- Image: {len(imagetestset)}")
    print(f"- Video: {len(videotestset)}")


    for item in texttestset:
        if item not in item2idx:
            print('MISSING:', item)

    EVAL_PAIRS = [
        (texttestset, texttestset, "Text->Text"),
        (videotestset, texttestset, "Video->Text"),
        (imagetestset, texttestset, "Image->Text"),
        (texttestset + imagetestset + videotestset, texttestset, "All->Text"),
        (videotestset, videotestset, "Video->Video"),
        (texttestset, videotestset, "Text->Video"),
        (imagetestset, videotestset, "Image->Video"),
        (texttestset + imagetestset + videotestset, videotestset, "All->Video"),
        (imagetestset, imagetestset, "Image->Image"),
        (texttestset, imagetestset, "Text->Image"),
        (videotestset, imagetestset, "Video->Image"),
        (texttestset + imagetestset + videotestset, imagetestset, "All->Image")
    ]

    for queryset, targetset, label in EVAL_PAIRS:
        scores = fx_calc_map_label_detailed(
            db_items=targetset,
            query_items=queryset,
            dist=predicted_dist,
            ground_dist=ground_dist,
            k=0,
            threshold=THRESHOLD,
            normalize_ap=NORMALIZE_AP,
            item2idx=item2idx,
            # save_pr_curve_flag=SAVE_PR,
            # pr_curve_dir=PR_DIR,
            # label=label.replace('->', '_')
        )
        print(f"{label}: MAP = {scores}")

    # if save_pr_curve_flag and pr_curve_dir and label:
    #         recall_levels, mean_precision = compute_interpolated_pr_curve(precision, recall)
    #         save_pr_curve(pr_curve_dir, label, recall_levels, mean_precision)


    dist = predicted_dist
    maps = []
    maps.append(cmac1.fx_calc_map_label_detailed(texttestset, texttestset, dist, ground_dist, item2idx, k = 0))         # !!
    maps.append(cmac1.fx_calc_map_label_detailed(videotestset, texttestset, dist, ground_dist, item2idx, k = 0))
    maps.append(cmac1.fx_calc_map_label_detailed(imagetestset, texttestset, dist, ground_dist, item2idx, k = 0))
    maps.append(cmac1.fx_calc_map_label_detailed(testset, texttestset, dist, ground_dist, item2idx, k = 0))
    #
    maps.append(cmac1.fx_calc_map_label_detailed(videotestset, videotestset, dist, ground_dist, item2idx, k = 0))
    maps.append(cmac1.fx_calc_map_label_detailed(texttestset, videotestset, dist, ground_dist, item2idx, k = 0)) #
    maps.append(cmac1.fx_calc_map_label_detailed(imagetestset, videotestset, dist, ground_dist, item2idx, k = 0))
    maps.append(cmac1.fx_calc_map_label_detailed(testset, videotestset, dist, ground_dist, item2idx, k = 0))
    #
    maps.append(cmac1.fx_calc_map_label_detailed(imagetestset, imagetestset, dist, ground_dist, item2idx, k = 0))
    maps.append(cmac1.fx_calc_map_label_detailed(texttestset, imagetestset, dist, ground_dist, item2idx, k = 0))  #
    maps.append(cmac1.fx_calc_map_label_detailed(videotestset, imagetestset, dist, ground_dist, item2idx, k = 0))   #
    maps.append(cmac1.fx_calc_map_label_detailed(testset, imagetestset, dist, ground_dist, item2idx, k = 0))

    print(np.mean(maps))
