import math
import pickle
import scipy
import numpy as np
import set_train_test_data_and_distance_matrix_from_mmir_ground_w_localDB as stt
# from set_train_test_data_from_mmir_ground import prepare_dataset
import os
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score, precision_recall_curve

with open("predicted_test_distance_matrix.pkl", "rb") as f:
    dist = pickle.load(f)
    print(np.shape(dist))
    
#dist = #scipy.spatial.distance.cdist(test, train, metric) # test is query, train is retrieval set, hahahah
#print(dist[1].reshape(914,2).tolist())
# ord = dist.argsort(1) # Returns the indices that would sort this array.  ######################### imp
#print(dist[1].reshape(914,2).tolist())
# print(ord)

scores = {} # scores is an empty dict already

if os.path.getsize("testset.pkl") > 0:
    with open("testset.pkl", "rb") as f:
        testset = pickle.load(f)
else:
    print("size 0")


if os.path.getsize("trainset.pkl") > 0:
    with open("trainset.pkl", "rb") as f:
        trainset = pickle.load(f)
else:
    print("size 0")

if os.path.getsize("validset.pkl") > 0:
    with open("validset.pkl", "rb") as f:
        validset = pickle.load(f)
else:
    print("size 0")

with open("testsetitem2idx.pkl", "rb") as f:
    item2idx = pickle.load(f)
with open("testsetidx2item.pkl", "rb") as f:
    idx2item = pickle.load(f)

with open("test_distance_matrix.pkl", "rb") as f:
    ground_dist = pickle.load(f)
    print(np.shape(ground_dist))

def calPrecision(y1, x1, y2, x2, x):
    if x2 == x1:
        return 0.0;
    result = (y2-y1)*(x-x1)*1.0/(x2-x1) + y1;
    return result

def sklearn_calc_map_and_PR(db_items, query_items, dist, ground_dist, k):
    numcases = len(db_items)
    query_indices = [item2idx[item] for item in query_items]
    db_indices = [item2idx[item] for item in db_items]
    trunc_dist = dist[np.ix_(query_indices, db_indices)]  # 6 query, so 6 rows, 3 content DB, so 3 columns
    trunc_gt = ground_dist[np.ix_(query_indices, db_indices)]
    AP = np.zeros((len(query_indices), 1))

    y_true = np.zeros(np.shape(trunc_gt))
    y_pred = np.zeros(np.shape(trunc_dist))

    # print(np.shape(y_true) == np.shape(y_pred))
    #
    # precision = np.zeros((len(query_indices), len(db_indices)))
    # recall = np.zeros((len(query_indices), len(db_indices)))
    for i, mapped_i in enumerate(query_indices):
        y_true[i, :] = [item < 3.0 for item in trunc_gt[i]]
        y_pred[i, :] = [item < 3.0 for item in trunc_dist[i]]

        AP[i] = average_precision_score(y_true[i], y_pred[i])

    return (np.mean(AP))


def fx_calc_map_label_detailed(db_items, query_items, dist, ground_dist, k):
    numcases = len(db_items)  # train_labels.shape[0]
    query_indices = [item2idx[item] for item in query_items]
    db_indices = [item2idx[item] for item in db_items]
    trunc_dist = dist[np.ix_(query_indices, db_indices)]  # 6 query, so 6 rows, 3 content DB, so 3 columns

    trunc_gt = ground_dist[np.ix_(query_indices, db_indices)]
    # assert np.array_equal(dist, dist_temp)

    ord = trunc_dist.argsort(1)  # Returns the indices that would sort this array.

    if k == 0:
        k = numcases
    #####################
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]                # [ len(db_items) ] / [539]

    # ks = [numcases] = [say, 500 items in DB]
    # calMAP called with _k=500
    #
    def calMAP(_k):     # @k??
        _res = []
        recall = np.zeros( (len(query_indices), len(db_indices)) );
        precision = np.zeros( (len(query_indices), len(db_indices)) );

        non_zero_indices = list()
        for i, mapped_i in enumerate(query_indices):
            predicted_order = ord[i]
            relAll = sum([item < 3.0 for item in trunc_gt[i]])      # Total#OfRelevantDocuments in gt
            if relAll == 0:
                print("zero")
                continue
            else:
                non_zero_indices.append(i)

            # iterate start through all db items
            p = 0.0
            r = 0.0
            total_predicted_relevant = 0
            for j in range(_k):             # iterate through all databases => as k = numcases = len(db_items)
                if trunc_dist[i][predicted_order[j]] < 3 : # predicted labels & ground labels  // this is my test label for query i
                    total_predicted_relevant += 1                           # inc it whenever this is a 'relevant' prediction
                    if trunc_gt[i][predicted_order[j]] < 3:
                        r += 1  # CorrectPredictions += 1
                        p += (r / (j + 1))  # CorrectPredictions divided by k

                precision[i, j] = (r / total_predicted_relevant) # (j+1)) #
                recall[i, j] = (r / (relAll * 1.0))

            # iterate END through all db items
            # So after this loop,
            # r: all correctly predicted samples for this query / CorrectPredictions / true positives
            # relAll: in original ground truth, how many samples has label 'relevant'
            # at each step in the loop (j+1) : was the value of k
            if r > 0:
                _res += [p / r]              # AP = RunningSum / Total#ofRelevantDocumentsInGT
            else:
                _res += [0]

        # print(res)                             # size of db_items
        map_score = np.mean(_res)
        # print("MAP: ", map_score)
        '''
        # the precision and recall array above are being used here
        recall = np.insert(recall, 0, 0, axis=1)
        recall = np.insert(recall, len(db_indices)+1, 1,  axis=1)                    # added before len()+1 position
        precision = np.insert(precision, 0, 1, axis=1)
        precision = np.insert(precision, len(db_indices)+1, 0, axis=1)
        precisionValue = np.zeros((1000, len(non_zero_indices)));       # For PR-curve, 1000 points on x-axis; ignores the one with no 'relevant' label in gt
        count = 0;
        for recallValue in np.arange(0.001, 1, 0.001):                  # 0.001 with 0.001 step until 1: 1000 iterations
            count += 1;

            # 6×7 logical array
                                                    # compare all query cases, hence : in column 0
            flag = (recall[:, 0:] < recallValue);   # don't compare the first column, that's just filler, hence 1: in column1
            flagPlace = np.sum(flag, 0);  # just changing axis to 1 from 0 is working miracles!! # changed back to 0 - solved error!

            # just got it, why they added 0 and 1 infront of recall and precision, as at 0th threshold, recall is 0,
            # precision is 1 in pr curve, thats why we would do range(1, _k)
            #
            for idx, j in enumerate(non_zero_indices):      # idx and j have same value, so same thing
                precisionValue[count, idx] = calPrecision(precision[j, flagPlace[j]], recall[j, flagPlace[j]],
                                                        precision[j, flagPlace[j]+1], recall[j, flagPlace[j]+1],
                                                        recallValue);

        recallValue = np.transpose([*np.arange(0.001, 1, 0.001)])           # needs new variable name, as used again
        # as count starts from 1, first one is always 0, if count start from 1, last one is always 0, instead just start
        # from index 1, then matches with recall calue len
        precision = np.mean(precisionValue[0:], 1);
        #####################################################
        '''

        ####
        precisionValue = np.zeros((len(non_zero_indices), 11))
        for query_no in non_zero_indices:
            count = 0
            for recallLevel in np.arange(0, 1.1, 0.1):  # 0.001 with 0.001 step until 1: 1000 iterations
                # For instance for recall level of 0.2, we need to find the maximum precision where recall is greater than
                # or equal to 0.2 in original recall precision table.
                indices_w_greater_recall = [idx for idx, v in enumerate(recall[query_no]) if v > recallLevel]
                precisions = [precision[query_no][idx] for idx in indices_w_greater_recall]
                if len(precisions) != 0:
                    precisionValue[query_no][count] = max(precisions)
                # else precisionValue[query_no][recallLevel] = 0 // already initialized
                count += 1
        # for recallLevel in np.arange(0, 1.1, 0.1):  # 0.001 with 0.001 step until 1: 1000 iterations
        #     count += 1
        #     # For instance for recall level of 0.2, we need to find the maximum precision where recall is greater than
        #     # or equal to 0.2 in original recall precision table.
        #     query_no = 0
        #     boolean_w_greater_recall = recall[:, :] > recallLevel
        #     indices_w_greater_recall = boolean_w_greater_recall.nonzero()
        #     print(indices_w_greater_recall)
        #     precisions = precision[np.ix_(indices_w_greater_recall)]
        #     print(precisions)
        #     # if len(precisions) != 0:
        #     #     precisionValue[query_no][count] = max(precisions)
        #     # # else precisionValue[query_no][recallLevel] = 0 // already initialized
        #     # print(precisionValue[query_no])
        recallValue = np.transpose([*np.arange(0, 1.1, 0.1)])
        precision = np.mean(precisionValue, 0)

        #######
        # precision = np.mean(precision, 0)
        # recallValue = np.mean(recall, 0)

#        # pr_display = PrecisionRecallDisplay(precision=precision, recall=recallValue).plot()
#        # plt.show()            # this shit was causing seg fault
#        # plt.close()

        # for safety commenting out write
        # with open("prcurve/" + str(len(query_indices)) + "-" + str(len(db_indices)) + "-recall-11-point.pkl", "wb") as f:
        #     pickle.dump(recallValue, f, protocol=2)
        # with open("prcurve/" + str(len(query_indices)) + "-" + str(len(db_indices)) + "-precision-11-point.pkl", "wb") as f:
        #     pickle.dump(precision, f, protocol=2)

        print(str(len(query_indices)) + "-" + str(len(db_indices)) + ": ", end=" ")

        return map_score # np.mean(_res)            # AP and mean(_res) is same

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

# dist will change based on what model we test  - SimGNN, SDML or SQL.
# ground_dist will always remain same
# subset of dist is selected in the fx_calc_map_label() function, that is to select subset of items for image/text/video subset test
#############################
# np.random.seed(1234)
# randtest = np.random.choice(testset, 10)
# randtrain = np.random.choice(testset, 3)
#############################
#print(fx_calc_map_label(testset, testset, dist, ground_dist, k = -1))
#print(fx_calc_map_label(randtrain, randtest, dist, ground_dist, k = 0)) #-1 sets _k to 50
# input()
rows = stt.prepare_dataset()
print("############# Start of size of the dataset #################")
texttestset = [row[5] for row in rows if row[5]]
imagetestset = [row[3] for row in rows if row[3]]
videotestset = [row[4] for row in rows if row[4]]
print("test-text:", len(texttestset))
print(len(imagetestset))
print(len(videotestset))

texttrainset = [row[8] for row in rows if row[8]]
imagetrainset = [row[6] for row in rows if row[6]]
videotrainset = [row[7] for row in rows if row[7]]
print("train-text:", len(texttrainset))
print(len(imagetrainset))
print(len(videotrainset))
textvalidset = [row[11] for row in rows if row[11]]
imagevalidset = [row[9] for row in rows if row[9]]
videovalidset = [row[10] for row in rows if row[10]]
print("valid-text:", len(textvalidset))
print(len(imagevalidset))
print(len(videovalidset))

print("############# Start of calc. MAP #################")
# print(sklearn_calc_map_and_PR(texttestset, texttestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(videotestset, texttestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(imagetestset, texttestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(testset, texttestset, dist, ground_dist, k = 0))
# print("########")
# #
# print(sklearn_calc_map_and_PR(videotestset, videotestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(texttestset, videotestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(imagetestset, videotestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(testset, videotestset, dist, ground_dist, k = 0))
# print("########")
# #
# print(sklearn_calc_map_and_PR(imagetestset, imagetestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(texttestset, imagetestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(videotestset, imagetestset, dist, ground_dist, k = 0))
# print(sklearn_calc_map_and_PR(testset, imagetestset, dist, ground_dist, k = 0))
print("########")
# #
print(fx_calc_map_label_detailed(texttestset, texttestset, dist, ground_dist, k = 0))         # !!
print(fx_calc_map_label_detailed(videotestset, texttestset, dist, ground_dist, k = 0))
print(fx_calc_map_label_detailed(imagetestset, texttestset, dist, ground_dist, k = 0))
print(fx_calc_map_label_detailed(testset, texttestset, dist, ground_dist, k = 0))
#
print(fx_calc_map_label_detailed(videotestset, videotestset, dist, ground_dist, k = 0))
print(fx_calc_map_label_detailed(texttestset, videotestset, dist, ground_dist, k = 0)) #
print(fx_calc_map_label_detailed(imagetestset, videotestset, dist, ground_dist, k = 0))
print(fx_calc_map_label_detailed(testset, videotestset, dist, ground_dist, k = 0))
#
print(fx_calc_map_label_detailed(imagetestset, imagetestset, dist, ground_dist, k = 0))
print(fx_calc_map_label_detailed(texttestset, imagetestset, dist, ground_dist, k = 0))  #
print(fx_calc_map_label_detailed(videotestset, imagetestset, dist, ground_dist, k = 0))   #
print(fx_calc_map_label_detailed(testset, imagetestset, dist, ground_dist, k = 0))