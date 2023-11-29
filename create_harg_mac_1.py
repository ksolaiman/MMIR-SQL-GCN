import math
import pickle
import scipy
import numpy as np
# import set_train_test_data_and_distance_matrix_from_mmir_ground_w_localDB as stt
# from set_train_test_data_from_mmir_ground import prepare_dataset
import os
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, precision_recall_curve
from munkres import Munkres, print_matrix, make_cost_matrix, DISALLOWED
# import psycopg2
# from psycopg2.extensions import AsIs
import json

# # set-up a postgres connection
# conn = psycopg2.connect(database='Salvi', user='Salvi', password='sholock',
#                             host='localhost', port=5433)
# dbcur = conn.cursor()
# print("connection successful")
def get_properties(mgid):
    sql = dbcur.mogrify("""SELECT mgid, ubc, lbc, gender from mmir_predicted where mgid=%s""", (AsIs(mgid),))
    dbcur.execute(sql)
    rows = dbcur.fetchall()
    return rows[0]
def create_harg(json_data, query_obj_prop, test_obj_prop, file_no, isTrain='train'):
    # for now lets keep it simple, later we can improve it
    # {"graph_1": [[0, 1], [1, 2], [2, 3], [3, 4]],
    #  "graph_2":  [[0, 1], [1, 2], [1, 3], [3, 4], [2, 4]],
    #  "labels_1": [2, 2, 2, 2],
    #  "labels_2": [2, 3, 2, 2, 2],
    #  "ged": 1}
    json_data['graph_1'] = [[0, 1], [1, 2], [1, 3], [1, 4]]
    json_data['graph_2'] = [[0, 1], [1, 2], [1, 3], [1, 4]]
    json_data['labels_1'] = [50, 51, query_obj_prop[1], query_obj_prop[2], query_obj_prop[3]]  # 50: root, 51: person
    json_data['labels_2'] = [50, 51, test_obj_prop[1], test_obj_prop[2], test_obj_prop[3]]

    if isTrain=='train':
        savefile = 'train_pairs'
    elif isTrain=='valid':
        savefile = 'valid_pairs'
    else:
        savefile = 'test_pairs'

    with open("../data/"+savefile+"/"+ str(file_no)+".json", "w") as outfile:
        json.dump(json_data, outfile)
def calc_munkres(db_items, query_items, train_dist, ground_train_dist, item2idx, mgid2predicted_properties_dict, isTrain):
    numcases = len(db_items)  # train_labels.shape[0]
    query_indices = [item2idx[item] for item in query_items]
    db_indices = [item2idx[item] for item in db_items]
    trunc_train_dist = train_dist[np.ix_(query_indices, db_indices)]
    trunc_train_gt_dist = ground_train_dist[np.ix_(query_indices, db_indices)]
    # print(query_indices)            # [1554, 2898, 3673, 1439, 1215, 2495, 4117, 2026, 1501, 4946, 711]
    # item2idx is for dist only

    # query_items[10] is 6113, no idx2item for it
    # print(item2idx[query_items[10]])            # 10 - 711

    trunc_train_prob = np.ones(np.shape(trunc_train_dist)) * math.inf
    ###############
    m = Munkres()
    sample_count = 0
    for idx1, item1 in enumerate(query_items):
        print(item1)
        # query_obj_properties = get_properties(item1)
        query_obj_properties = mgid2predicted_properties_dict[item1]
        for idx2, item2 in enumerate(db_items):
            if item1 != item2:                      # if they are same item munkres cant calculate ofcourse
                if isTrain:
                    # use munkres to calculate
                    # Step 1: create cost-matrix
                    # For now, just putting it directly, but need a func to create the matrix
                    # considering trunc_train_dist[idx1][idx2] as eplv cost
                    '''
                    matrix = [[trunc_train_dist[idx1][idx2], math.inf],
                              [math.inf, trunc_train_dist[idx1][idx2]]]
                    # Step 2: Run the algo
                    indexes = m.compute(matrix)
                    row, column = indexes[0]            # just need the root cost, as cum. Munkres would have total cose in root
                    calc_dist = matrix[row][column]
                    '''
                    # total = 0
                    # for row, column in indexes:
                    #     value = matrix[row][column]
                    #     total += value
                    # print(total)
                    # trunc_train_prob = np.exp((-1) * calc_dist / 4)
                ####
                # create both graphs - json for SimGNN
                db_obj_properties = mgid2predicted_properties_dict[item2]
                # db_obj_properties = get_properties(item2)

                ###
                json_data = {}
                json_data['ged'] = trunc_train_dist[idx1][idx2]             # should be calc dist - using prop. from identifiers
                if isTrain=='valid':
                    json_data['ged'] = trunc_train_gt_dist[idx1][idx2]      # pass the real ged as target for this
                # json_data['gt_ged'] = trunc_train_gt_dist[idx1][idx2]       # should be calc dist - using gold properties
                # json_data['query_mgid'] = int(item1)
                # json_data['dbitem_mgid'] = int(item2)
                create_harg(json_data, query_obj_properties,                      # query  mgid
                                db_obj_properties, sample_count, isTrain)                      # test mgid
                sample_count += 1
    ###############
    # trunc_train_prob = np.exp((-1) * trunc_train_dist / 4)  # as g_q and g_c is of size 4; divided by (4+4/2)
    # print(trunc_train_prob)
    #

    #closing database connection.
    # if(conn):
    #     dbcur.close()
    #     conn.close()
    #     print("PostgreSQL connection is closed")


###### calc predicted dist and write ##
# pr_train_dist, item2idx, idx2item = stt.calc_distance_local_DB_predicted(trainset)
# stt.write_dataset(trainset, item2idx, idx2item, 'trainset2')
# with open("predicted_train_distance_matrix.pkl", "wb") as f:
#     pickle.dump(pr_train_dist, f)

def train_create_graph(trainOrValid):
    # trainset = [row[0] for row in rows]
    if trainOrValid=='valid':
        filename1='trainset'        # correct: both valid and train should read trainsetmgid2predicted_prop_dict as full trainset was load in it
    else:
        filename1='trainset'
    with open(filename1+".pkl", "rb") as f:
        trainset = pickle.load(f)
    with open(filename1+"item2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)

    with open("predicted_train_distance_matrix.pkl", "rb") as f:
        predicted_train_dist = pickle.load(f)

    with open("train_distance_matrix.pkl", "rb") as f:
        ground_train_dist = pickle.load(f)

    # mgid2predicted_properties_dict = {}
    # for idx1, item1 in enumerate(trainset):
    #     mgid2predicted_properties_dict[item1] = get_properties(item1)
    # with open(filename1+"mgid2predicted_prop_dict.pkl", "wb") as f:
    #     pickle.dump(mgid2predicted_properties_dict, f)
    with open(filename1+"mgid2predicted_prop_dict.pkl", "rb") as f:
        mgid2predicted_properties_dict = pickle.load(f)

    np.random.seed(42)
    if trainOrValid== 'train':
        calc_munkres(np.random.choice(trainset[0:3000], 200, False), np.random.choice(trainset[0:3000], 100, False),
                     predicted_train_dist,
                     ground_train_dist, item2idx, mgid2predicted_properties_dict, trainOrValid)
    else:
        calc_munkres(np.random.choice(trainset[3000:], 100, False), np.random.choice(trainset[3000:], 100, False), predicted_train_dist,
                 ground_train_dist, item2idx, mgid2predicted_properties_dict, trainOrValid)


def test():
    testset = [row[2] for row in rows if row[2]]
    with open("testset.pkl", "rb") as f:
        testset2 = pickle.load(f)
    assert  testset == testset2
    print(testset[0:5])
    print(testset2[0:5])
    with open("testsetitem2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)
    with open("testsetidx2item.pkl", "rb") as f:
        idx2item = pickle.load(f)
    #
    with open("predicted_test_distance_matrix.pkl", "rb") as f:
        predicted_test_dist = pickle.load(f)
    #
    with open("test_distance_matrix.pkl", "rb") as f:
        ground_test_dist = pickle.load(f)
    #
    # calc_munkres(testset, testset, predicted_test_dist, ground_test_dist, item2idx, 'test')

    # mgid2predicted_properties_dict = {}
    # for idx1, item1 in enumerate(testset):
    #     # query_obj_properties = get_properties(item1)
    #     mgid2predicted_properties_dict[item1] = get_properties(item1)
    # with open("mgid2predicted_prop_dict.pkl", "wb") as f:
    #     pickle.dump(mgid2predicted_properties_dict, f)
    with open("mgid2predicted_prop_dict.pkl", "rb") as f:
        mgid2predicted_properties_dict = pickle.load(f)


if __name__ == "__main__":
    # rows = stt.prepare_dataset()
    train_create_graph('valid')
    # test()

#
# print("############# Start of size of the dataset #################")
# texttestset = [row[5] for row in rows if row[5]]
# imagetestset = [row[3] for row in rows if row[3]]
# videotestset = [row[4] for row in rows if row[4]]
# print("test-text:", len(texttestset))
# print(len(imagetestset))
# print(len(videotestset))
#
# texttrainset = [row[8] for row in rows if row[8]]
# imagetrainset = [row[6] for row in rows if row[6]]
# videotrainset = [row[7] for row in rows if row[7]]
# print("train-text:", len(texttrainset))
# print(len(imagetrainset))
# print(len(videotrainset))
# textvalidset = [row[11] for row in rows if row[11]]
# imagevalidset = [row[9] for row in rows if row[9]]
# videovalidset = [row[10] for row in rows if row[10]]
# print("valid-text:", len(textvalidset))
# print(len(imagevalidset))
# print(len(videovalidset))
#
