import psycopg2
import numpy as np
import pickle
import json
import os
import utils

# Load config file
config = utils.load_config()
which_db = config.get('db_host')
db_settings = config.get(which_db)

def calc_distance_local_DB(testset, dbcur=None):
    '''
    testset: a list containing the id of the mmir obj. in mmir_ground and mmir_predicted table
    '''
    item2idx = dict()
    idx2item = dict()
    for idx, item in enumerate(testset):
        item2idx[item]=idx
        idx2item[idx]=item
        
    dist = np.zeros((len(testset), len(testset)))
    predicted_dist = np.zeros((len(testset), len(testset)))
    print(dist.shape)
    
    for i in range(0, len(testset)):
        for j in range(0, len(testset)):
            if i == j:
                dist[i][j] = np.inf
                predicted_dist[i][j] = np.inf

    print(f"Connecting to {which_db} database at {db_settings['host']}:{db_settings['port']}")

    if dbcur is None:
        conn = psycopg2.connect(**db_settings)
        dbcur = conn.cursor()
        print("connection successful")
    try:
        sql = dbcur.mogrify("""
        SELECT 
                A.mgid as Aid, B.mgid as Bid,
                CONCAT(A.ubc::character varying, A.lbc::character varying, 
        A.gender::character varying) as Alabel,
        CONCAT(B.ubc::character varying, B.lbc::character varying, 
        B.gender::character varying) as Blabel,
        (CASE WHEN A.ubc=B.ubc THEN 0
                            ELSE 1
                    END) + 
                (CASE WHEN A.gender=B.gender THEN 0
                            ELSE 3
                    END) + 
                (CASE WHEN A.lbc=B.lbc THEN 0
                            ELSE 2
                    END)
                AS distance
            FROM mmir_ground A
            JOIN mmir_ground B 
            ON 
                A.mgid != B.mgid
                and 
                A.ubc!= -1 and A.lbc!= -1 and A.ubc!= 100 and A.lbc!= 100
                and 
                B.ubc!= -1 and B.lbc!= -1 and B.ubc!= 100 and B.lbc!= 100
                and A.mgid in %s --(6, 2)
                and B.mgid in %s --(5, 3)
        --order by A.mgid, distance asc

        --Limit 1000
        """, (tuple(testset), tuple(testset))) #(tuple(testset[0:3]), tuple(testset[2:5])))
        dbcur.execute(sql)
        rows = dbcur.fetchall()
        
        for item in rows:
            # print(item)
            dist[item2idx[item[0]]][item2idx[item[1]]] = item[4]


        sql = dbcur.mogrify("""
        SELECT 
                A.mgid as Aid, B.mgid as Bid,
                CONCAT(A.ubc::character varying, A.lbc::character varying, 
        A.gender::character varying) as Alabel,
        CONCAT(B.ubc::character varying, B.lbc::character varying, 
        B.gender::character varying) as Blabel,
        (CASE WHEN A.ubc=B.ubc THEN 0
                            ELSE 1
                    END) + 
                (CASE WHEN A.gender=B.gender THEN 0
                            ELSE 3
                    END) + 
                (CASE WHEN A.lbc=B.lbc THEN 0
                            ELSE 2
                    END)
                AS distance
            FROM mmir_predicted A
            JOIN mmir_predicted B 
            ON 
                A.mgid != B.mgid
                and 
                A.ubc!= -1 and A.lbc!= -1 and A.ubc!= 100 and A.lbc!= 100
                and 
                B.ubc!= -1 and B.lbc!= -1 and B.ubc!= 100 and B.lbc!= 100
                and A.mgid in %s --(6, 2)
                and B.mgid in %s --(5, 3)
        --order by A.mgid, distance asc

        --Limit 1000
        """, (tuple(testset), tuple(testset)))  # (tuple(testset[0:3]), tuple(testset[2:5])))
        dbcur.execute(sql)
        rows = dbcur.fetchall()

        for item in rows:
            predicted_dist[item2idx[item[0]]][item2idx[item[1]]] = item[4]

    except (Exception, psycopg2.Error) as error :
        print ("Error while fetching data from PostgreSQL", error)

    finally:
        #closing database connection.
        if(conn):
            dbcur.close()
            conn.close()
            print("PostgreSQL connection is closed")
    
    return dist, predicted_dist, item2idx, idx2item

def prepare_dataset(random_seed=0.1234):

    print(f"Connecting to {which_db} database at {db_settings['host']}:{db_settings['port']}")

    # Connect using the selected settings
    conn = psycopg2.connect(**db_settings)
    dbcur = conn.cursor()
    print("connection successful")
    try:
        dbcur.execute("set seed to %s;", (random_seed,))    # had to separate as now it takes argument and 
                                # mutliple line argument parsing is not great for sql query
        sql = dbcur.mogrify("""
        -- set seed to %s;
        select --unnest(itest), unnest(itrain) --, 
        --itrain || vtrain || ttrain as trainsetByClass, array_length(itrain || vtrain || ttrain, 1), label
        unnest(itrain || vtrain || ttrain) as trainset,
        unnest(ivalid || vvalid || tvalid) as validset,
        unnest(itest || vtest || ttest) as testset,
        unnest(itest) as imgtestset,
        unnest(vtest) as videotestset,
        unnest(ttest) as texttestset,
        unnest(itrain) as imgtrainset,
        unnest(vtrain) as videotrainset,
        unnest(ttrain) as texttrainset,
        unnest(ivalid) as imgvalidset,
        unnest(vvalid) as videovalidset,
        unnest(tvalid) as textvalidset
    --ivalid || vvalid || tvalid as validset
    --sum(array_length(vtest, 1)) + sum(array_length(itest, 1)) + sum(array_length(ttest, 1)) as test_size
    --, 
    --sum(itrainlen), sum(ivalidlen)
    -- each one of the upper_command is right, just gives different results
    from
    (select *,
    array_length(imageids, 1),
    array_length(videoids, 1),
    imageids[0: floor(array_length(imageids, 1)*0.6)] as itrain,
    array_length(imageids[0: floor(array_length(imageids, 1)*0.6)], 1) as itrainlen,
    imageids[floor(array_length(imageids, 1)*0.6)+1:floor(array_length(imageids, 1)*0.8)] as ivalid,
    array_length(imageids[floor(array_length(imageids, 1)*0.6)+1:floor(array_length(imageids, 1)*0.8)], 1) as ivalidlen,
    imageids[floor(array_length(imageids, 1)*0.8)+1:] as itest,
    array_length(imageids[floor(array_length(imageids, 1)*0.8)+1:], 1) as itestlen,

    textids[0: floor(array_length(textids, 1)*0.6)] as ttrain,
    textids[floor(array_length(textids, 1)*0.6)+1:floor(array_length(videoids, 1)*0.8)] as tvalid,
    textids[floor(array_length(textids, 1)*0.8)+1:] as ttest,

    videoids[0: floor(array_length(videoids, 1)*0.6)] as vtrain,
    videoids[floor(array_length(videoids, 1)*0.6)+1:floor(array_length(videoids, 1)*0.8)] as vvalid,
    videoids[floor(array_length(videoids, 1)*0.8)+1:] as vtest
    from (
        select 
        string_agg(mgid::varchar, ', ') as ids, 
        -- string_agg(ctype::character varying, ', ') as ctypes,
        COUNT(CASE WHEN ctype='image' THEN 1 END) as nimage,
        COUNT(CASE WHEN ctype='video' THEN 1 END) as nvideo,
        COUNT(CASE WHEN ctype='untext' THEN 1 END) as ntext,
        array_remove(array_agg(CASE WHEN ctype='image' 
                               THEN mgid END 
                               order by random()), NULL) as imageids,
        array_remove(array_agg(CASE WHEN ctype='image' 
                               THEN mgid END), NULL) as imageids2,
        array_remove(array_agg(CASE WHEN ctype='video' 
                               THEN mgid END
                               order by random()), NULL) as videoids,
        array_remove(array_agg(CASE WHEN ctype='untext' 
                               THEN mgid END
                               order by random()), NULL) as textids,
        ubc, lbc, gender, 
        count(*) as freq,
        CONCAT(ubc::character varying, lbc::character varying, 
               gender::character varying) as label
        from mmir_ground
        where ubc!= -1 and lbc!= -1 and ubc!= 100 and lbc!= 100 
        -- for protection against text attributes
        -- and ctype='untext'
        group by ubc, lbc, gender
        order by freq desc
    )fullTable
    where freq >= 10
    -- and nimage != 0 and nvideo != 0 and ntext != 0
     ) idstable
        """)
        dbcur.execute(sql)
        rows = dbcur.fetchall()
    
    except (Exception, psycopg2.Error) as error :
        print ("Error while fetching data from PostgreSQL", error)

    finally:
        #closing database connection.
        if(conn):
            dbcur.close()
            conn.close()
            print("PostgreSQL connection is closed")

    return rows 


def save_dist_matrices(split_name, ground_dist, predicted_dist, item2idx, idx2item, base_dir="dataset"):
    """
    Saves the distance matrices and item-index mappings in the structured folder:
    
    dataset/
      dist_matrices/
        split_name/  (train/ or test/)
          item2idx.pkl
          idx2item.pkl
          ground_distance_matrix.pkl
          predicted_distance_matrix.pkl
    """
    import os
    import pickle

    # Define target folder
    target_dir = os.path.join(base_dir, "dist_matrices", split_name)
    os.makedirs(target_dir, exist_ok=True)

    # Save ground distance matrix
    with open(os.path.join(target_dir, "ground_distance_matrix.pkl"), "wb") as f:
        pickle.dump(ground_dist, f)

    # Save predicted distance matrix
    with open(os.path.join(target_dir, "predicted_distance_matrix.pkl"), "wb") as f:
        pickle.dump(predicted_dist, f)

    # Save item2idx mapping
    with open(os.path.join(target_dir, "item2idx.pkl"), "wb") as f:
        pickle.dump(item2idx, f)

    # Save idx2item mapping
    with open(os.path.join(target_dir, "idx2item.pkl"), "wb") as f:
        pickle.dump(idx2item, f)

    print(f"Saved distance matrices and mappings to {target_dir}")

def load_dist_matrices(split_name, base_dir="dataset"):
    """
    Loads distance matrices and item-index mappings for the given split (train or test)
    from the structured folder:
    
    dataset/
      dist_matrices/
        split_name/
          item2idx.pkl
          idx2item.pkl
          ground_distance_matrix.pkl
          predicted_distance_matrix.pkl
          
    Returns:
        (ground_dist, predicted_dist, item2idx, idx2item)
    """

    target_dir = os.path.join(base_dir, "dist_matrices", split_name)

    with open(os.path.join(target_dir, "ground_distance_matrix.pkl"), "rb") as f:
        ground_dist = pickle.load(f)

    with open(os.path.join(target_dir, "predicted_distance_matrix.pkl"), "rb") as f:
        predicted_dist = pickle.load(f)

    with open(os.path.join(target_dir, "item2idx.pkl"), "rb") as f:
        item2idx = pickle.load(f)

    with open(os.path.join(target_dir, "idx2item.pkl"), "rb") as f:
        idx2item = pickle.load(f)

    print(f"Loaded distance matrices and mappings from {target_dir}")

    return ground_dist, predicted_dist, item2idx, idx2item


# This is a legacy function, before I added load_dist_matrices(). Currently no use of it.
def read_dataset(name, filename, predicted_matrix_filename):
    with open(name+".pkl", "rb") as f:
        testset  = pickle.load(f)
    with open(name+"item2idx.pkl", "rb") as f:
        testsetitem2idx = pickle.load(f)
    with open(name+"idx2item.pkl", "rb") as f:
        testsetidx2item = pickle.load(f)
    with open(filename+".pkl", "rb") as f:
        distance_matrix = pickle.load(f)
    with open(predicted_matrix_filename+".pkl", "rb") as f:
        predicted_distance_matrix = pickle.load(f)

    return testset, testsetitem2idx, testsetidx2item, distance_matrix, predicted_distance_matrix

def prepare_train_pool_and_test_splits(rows=None, random_seed=0.1234):
    """
    Given DB rows with train, valid, test columns for each modality,
    combine train+valid into trainpool (for overall and per modality).

    If `rows` is None, it will call prepare_dataset() automatically.
    This function replaces older manual splitting code in main(),
    by cleanly generating the train pools and test sets ready to save.

    `random_seed`: seed to pass to database for the random shuffling in prepare_dataset(), 
            optional, default set to 0.1234

    Returns:
        Dictionary with keys:
            - trainpool, testset
            - image_trainpool, image_testset
            - text_trainpool, text_testset
            - video_trainpool, video_testset
    """
    if rows is None:
        rows = prepare_dataset(random_seed=random_seed)

    # Overall train/valid/test
    trainset = [row[0] for row in rows if row[0]]
    validset = [row[1] for row in rows if row[1]]
    testset = [row[2] for row in rows if row[2]]
    trainpool = trainset + validset

    # Image
    imgtrainset = [row[6] for row in rows if row[6]]
    imgvalidset = [row[9] for row in rows if row[9]]
    imagetestset = [row[3] for row in rows if row[3]]
    image_trainpool = imgtrainset + imgvalidset

    # Video
    videotrainset = [row[7] for row in rows if row[7]]
    videovalidset = [row[10] for row in rows if row[10]]
    videotestset = [row[4] for row in rows if row[4]]
    video_trainpool = videotrainset + videovalidset

    # Text
    texttrainset = [row[8] for row in rows if row[8]]
    textvalidset = [row[11] for row in rows if row[11]]
    texttestset = [row[5] for row in rows if row[5]]
    text_trainpool = texttrainset + textvalidset

    return {
        "trainpool": trainpool,
        "testset": testset,
        "image_trainpool": image_trainpool,
        "image_testset": imagetestset,
        "video_trainpool": video_trainpool,
        "video_testset": videotestset,
        "text_trainpool": text_trainpool,
        "text_testset": texttestset
    }

def save_train_test_splits(splits_dict, base_dir="dataset"):
    """
    Saves the trainpool and testset data for overall and per modality
    into a folder structure like:
    
    dataset/
        trainpool.pkl
        testset.pkl
        image/
            image_trainpool.pkl
            image_testset.pkl
        video/
            video_trainpool.pkl
            video_testset.pkl
        text/
            text_trainpool.pkl
            text_testset.pkl
    """
    os.makedirs(base_dir, exist_ok=True)

    # Save overall trainpool and testset
    with open(os.path.join(base_dir, "trainpool.pkl"), "wb") as f:
        pickle.dump(splits_dict["trainpool"], f)
    with open(os.path.join(base_dir, "testset.pkl"), "wb") as f:
        pickle.dump(splits_dict["testset"], f)

    # Image
    image_dir = os.path.join(base_dir, "image")
    os.makedirs(image_dir, exist_ok=True)
    with open(os.path.join(image_dir, "image_trainpool.pkl"), "wb") as f:
        pickle.dump(splits_dict["image_trainpool"], f)
    with open(os.path.join(image_dir, "image_testset.pkl"), "wb") as f:
        pickle.dump(splits_dict["image_testset"], f)

    # Video
    video_dir = os.path.join(base_dir, "video")
    os.makedirs(video_dir, exist_ok=True)
    with open(os.path.join(video_dir, "video_trainpool.pkl"), "wb") as f:
        pickle.dump(splits_dict["video_trainpool"], f)
    with open(os.path.join(video_dir, "video_testset.pkl"), "wb") as f:
        pickle.dump(splits_dict["video_testset"], f)

    # Text
    text_dir = os.path.join(base_dir, "text")
    os.makedirs(text_dir, exist_ok=True)
    with open(os.path.join(text_dir, "text_trainpool.pkl"), "wb") as f:
        pickle.dump(splits_dict["text_trainpool"], f)
    with open(os.path.join(text_dir, "text_testset.pkl"), "wb") as f:
        pickle.dump(splits_dict["text_testset"], f)

def load_all_splits(base_dir="dataset"):
    """
    Loads all trainpools and testsets from the dataset folder structure.
    
    Returns a dictionary with:
      - trainpool, testset
      - image_trainpool, image_testset
      - video_trainpool, video_testset
      - text_trainpool, text_testset
    """
    data = {}

    # Overall
    data['trainpool'] = utils.load_pickle(os.path.join(base_dir, "trainpool.pkl"))
    data['testset'] = utils.load_pickle(os.path.join(base_dir, "testset.pkl"))

    # Image
    image_dir = os.path.join(base_dir, "image")
    data['image_trainpool'] = utils.load_pickle(os.path.join(image_dir, "image_trainpool.pkl"))
    data['image_testset'] = utils.load_pickle(os.path.join(image_dir, "image_testset.pkl"))

    # Video
    video_dir = os.path.join(base_dir, "video")
    data['video_trainpool'] = utils.load_pickle(os.path.join(video_dir, "video_trainpool.pkl"))
    data['video_testset'] = utils.load_pickle(os.path.join(video_dir, "video_testset.pkl"))

    # Text
    text_dir = os.path.join(base_dir, "text")
    data['text_trainpool'] = utils.load_pickle(os.path.join(text_dir, "text_trainpool.pkl"))
    data['text_testset'] = utils.load_pickle(os.path.join(text_dir, "text_testset.pkl"))

    return data

def train_validation_split(trainpool, val_ratio=0.2, random_seed=42):
    """
    Random non-stratified train/validation split.
    Shuffles all IDs together and splits ignoring modality.
    """
    np.random.seed(random_seed)

    shuffled = np.random.permutation(trainpool)
    n_val = int(len(shuffled) * val_ratio)
    validation = shuffled[:n_val].tolist()
    train = shuffled[n_val:].tolist()

    return train, validation


def stratified_train_validation_split_by_modality(
    image_pool, video_pool, text_pool,
    balanced=False, undersample=True, oversample=False,
    SAMPLE_SIZE=2000,
    val_ratio=0.2, random_seed=42
):
    """
    Stratified train/validation split maintaining modality ratios.
    """
    np.random.seed(random_seed)

    def sample(pool, size, replace):
        return np.random.choice(pool, size=size, replace=replace).tolist()

    if balanced:
        if undersample:
            SAMPLE_SIZE = min(len(image_pool), len(video_pool), len(text_pool))
            replace_flags = (False, False, False)
        elif oversample:
            SAMPLE_SIZE = max(len(image_pool), len(video_pool), len(text_pool))
            replace_flags = (False, True, True)
        else:
            replace_flags = (False, True, True)

        image_train, image_val = train_validation_split(sample(image_pool, SAMPLE_SIZE, replace_flags[0]), val_ratio, random_seed)
        video_train, video_val = train_validation_split(sample(video_pool, SAMPLE_SIZE, replace_flags[1]), val_ratio, random_seed)
        text_train, text_val = train_validation_split(sample(text_pool, SAMPLE_SIZE, replace_flags[2]), val_ratio, random_seed)

    else:
        image_train, image_val = train_validation_split(image_pool, val_ratio, random_seed)
        video_train, video_val = train_validation_split(video_pool, val_ratio, random_seed)
        text_train, text_val = train_validation_split(text_pool, val_ratio, random_seed)

    combined_train = image_train + video_train + text_train
    combined_val = image_val + video_val + text_val

    return combined_train, combined_val, {
        "image_train": image_train,
        "image_val": image_val,
        "video_train": video_train,
        "video_val": video_val,
        "text_train": text_train,
        "text_val": text_val
    }


def k_fold_split(trainpool, k=5, random_seed=42):
    """
    Non-stratified k-fold splitting.
    Shuffles all IDs and splits ignoring modality.
    Yields (train, validation) for each fold.
    """
    np.random.seed(random_seed)

    shuffled = np.random.permutation(trainpool)
    fold_size = len(shuffled) // k
    folds = [shuffled[i*fold_size : (i+1)*fold_size].tolist() for i in range(k-1)]
    folds.append(shuffled[(k-1)*fold_size:].tolist())

    for i in range(k):
        val = folds[i]
        train = sum([f for j, f in enumerate(folds) if j != i], [])
        yield train, val

def stratified_k_fold_split_by_modality(
    image_pool, video_pool, text_pool,
    k=5, random_seed=42,
    balanced=False,
    undersample=True,
    oversample=False,
    SAMPLE_SIZE=None
):
    """
    Stratified k-fold splitting with optional modality balancing.
    Splits each modality pool into k folds separately.
    Yields (train, validation, details) per fold.
    """
    np.random.seed(random_seed)

    def sample(pool, size, replace):
        return np.random.choice(pool, size=size, replace=replace).tolist()

    def make_folds(pool):
        shuffled = np.random.permutation(pool)
        fold_size = len(shuffled) // k
        folds = [shuffled[i * fold_size: (i + 1) * fold_size].tolist() for i in range(k - 1)]
        folds.append(shuffled[(k - 1) * fold_size:].tolist())
        return folds

    # === Balance pools before fold splitting ===
    if balanced:
        if undersample:
            SAMPLE_SIZE = min(len(image_pool), len(video_pool), len(text_pool)) if SAMPLE_SIZE is None else SAMPLE_SIZE
            replace_flags = (False, False, False)
        elif oversample:
            SAMPLE_SIZE = max(len(image_pool), len(video_pool), len(text_pool)) if SAMPLE_SIZE is None else SAMPLE_SIZE
            replace_flags = (False, True, True)
        else:  # mixed
            SAMPLE_SIZE = SAMPLE_SIZE or min(len(image_pool), len(video_pool), len(text_pool))
            replace_flags = (False, True, True)

        image_pool = sample(image_pool, SAMPLE_SIZE, replace_flags[0])
        video_pool = sample(video_pool, SAMPLE_SIZE, replace_flags[1])
        text_pool = sample(text_pool, SAMPLE_SIZE, replace_flags[2])

    # === Perform fold splitting ===
    image_folds = make_folds(image_pool)
    video_folds = make_folds(video_pool)
    text_folds = make_folds(text_pool)

    for i in range(k):
        image_val = image_folds[i]
        video_val = video_folds[i]
        text_val = text_folds[i]

        image_train = sum([f for j, f in enumerate(image_folds) if j != i], [])
        video_train = sum([f for j, f in enumerate(video_folds) if j != i], [])
        text_train = sum([f for j, f in enumerate(text_folds) if j != i], [])

        combined_train = image_train + video_train + text_train
        combined_val = image_val + video_val + text_val

        yield combined_train, combined_val, {
            "image_train": image_train,
            "image_val": image_val,
            "video_train": video_train,
            "video_val": video_val,
            "text_train": text_train,
            "text_val": text_val
        }



# shows usage
def main():
    # preparing and writing data
    splits = prepare_train_pool_and_test_splits(random_seed=0.1237)
    save_train_test_splits(splits, base_dir=config.get('dataset_dir', "dataset")) 

    # Load everything once
    all_data = load_all_splits(config.get('dataset_dir', "dataset"))

    # Access overall pools
    trainpool = all_data['trainpool']
    testset = all_data['testset']

    # Dynamic split
    train, val = train_validation_split(trainpool, val_ratio=0.2, random_seed=42)

    print("Train size:", len(train))
    print("Validation size:", len(val))
    print("Test size (fixed):", len(testset))

    # Access modality-specific train pools
    image_pool = all_data['image_trainpool']
    video_pool = all_data['video_trainpool']
    text_pool = all_data['text_trainpool']

    # Access modality-specific test sets
    image_testset = all_data['image_testset']
    video_testset = all_data['video_testset']
    text_testset = all_data['text_testset']


    train, val, details = stratified_train_validation_split_by_modality(
        image_pool, video_pool, text_pool,
        val_ratio=0.2,
        random_seed=42
    )

    for i, (train, val) in enumerate(k_fold_split(trainpool, k=5, random_seed=42)):
        print(f"Fold {i+1}: Train={len(train)}, Val={len(val)}")

    for i, (train, val, details) in enumerate(
        stratified_k_fold_split_by_modality(
            image_pool, video_pool, text_pool,
            k=5,
            random_seed=42
        )
    ):
        print(f"Stratified Fold {i+1}: Train={len(train)}, Val={len(val)}")


    # # pre-calculate and save the distance matrices
    # dist, predicted_dist, item2idx, idx2item = calc_distance_local_DB(testset)
    # save_dist_matrices(
    #     split_name="test",
    #     ground_dist=dist,
    #     predicted_dist=predicted_dist,
    #     item2idx=item2idx,
    #     idx2item=idx2item, 
    #     base_dir=config.get("dataset_dir", "dataset")
    # )

    # dist, predicted_dist, item2idx, idx2item = calc_distance_local_DB(trainpool)
    # save_dist_matrices(
    #     split_name="train",
    #     ground_dist=dist,
    #     predicted_dist=predicted_dist,
    #     item2idx=item2idx,
    #     idx2item=idx2item, 
    #     base_dir=config.get("dataset_dir", "dataset")
    # )

    # load the distance matrices
    train_ground, train_predicted, train_item2idx, train_idx2item = load_dist_matrices("train", base_dir=config.get("dataset_dir", "dataset"))
    test_ground, test_predicted, test_item2idx, test_idx2item = load_dist_matrices("test", base_dir=config.get("dataset_dir", "dataset"))
    

if __name__ == "__main__":
    main()