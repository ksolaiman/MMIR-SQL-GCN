"""SimGNN runner."""

from utils import tab_printer
from simgnn import SimGNNTrainer
from param_parser import parameter_parser
import torch
import pickle
import numpy as np
from utils import process_pair, calculate_loss, calculate_normalized_ged

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    if not args.evaluate_only:
        trainer.fit()
    else:
        # trainer.score()
        trainer.model.load_state_dict(torch.load(args.load_path))
        trainer.model.eval()
        # create the json object basically # data = process_pair(graph_pair)

def fx_calc_map_label(db_indices, query_indices, trunc_dist, trunc_gt, k=0):
    numcases = len(db_indices)  # train_labels.shape[0]
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
        recall = np.zeros((len(query_indices), len(db_indices)));
        precision = np.zeros((len(query_indices), len(db_indices)));

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
                if trunc_dist[i][predicted_order[j]] < 3: # predicted labels & ground labels  // this is my test label for query i
                    total_predicted_relevant += 1                           # inc it whenever this is a 'relevant' prediction
                    if trunc_gt[i][predicted_order[j]] < 3:
                        r += 1  # CorrectPredictions += 1
                        p += (r / (j + 1))  # CorrectPredictions divided by k

                if total_predicted_relevant > 0:
                    precision[i, j] = (r / total_predicted_relevant) # (j+1)) #
                else:
                    precision[i, j] = 0
                recall[i, j] = (r / (relAll * 1.0))

            # iterate END through all db items
            # So after this loop,
            # r: all correctly predicted samples for this query / CorrectPredictions / true positives
            # relAll: in original ground truth, how many samples has label 'relevant'
            # at each step in the loop (j+1) : was the value of k
            if r > 0:
                _res += [p / (relAll * 1.0)]              # AP = RunningSum / Total#ofRelevantDocumentsInGT                     # previously was dividing by r, and was ok, bcoz the cond. was only 2nd one when gt was relevant, but now we have 2 conds. if prediction is relevant and gt is relevant
            else:
                _res += [0]

        # print(res)                             # size of db_items
        map_score = np.mean(_res)
        # print("MAP: ", map_score)

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
        recallValue = np.transpose([*np.arange(0, 1.1, 0.1)])
        precision = np.mean(precisionValue, 0)

        #### for safety commenting out write
        with open("./data/prcurve/" + str(len(query_indices)) + "-" + str(len(db_indices)) + "-recall-11-point-4_epoch_41__with_histogram.pkl", "wb") as f:
            pickle.dump(recallValue, f, protocol=2)
        with open("./data/prcurve/" + str(len(query_indices)) + "-" + str(len(db_indices)) + "-precision-11-point-4_epoch_41__with_histogram.pkl", "wb") as f:
            pickle.dump(precision, f, protocol=2)

        print(str(len(query_indices)) + "-" + str(len(db_indices)) + ": ", end=" ")

        return map_score                    # np.mean(_res)            # AP and mean(_res) is same

    res = []
    for k in ks:
        res.append(calMAP(k))
    print(res)
    return res

def calc_ced(db_items, query_items, train_dist, ground_train_dist, item2idx, mgid2predicted_properties_dict, trainer):
    numcases = len(db_items)  # train_labels.shape[0]
    query_indices = [item2idx[item] for item in query_items]
    db_indices = [item2idx[item] for item in db_items]
    trunc_train_dist = train_dist[np.ix_(query_indices, db_indices)]
    trunc_train_gt_dist = ground_train_dist[np.ix_(query_indices, db_indices)]

    # print(trunc_train_gt_dist)
    # print(trunc_train_dist)
    # input()

    dist = np.zeros((len(query_items), len(db_items)))

    for i in range(0, len(query_items)):
        for j in range(0, len(db_items)):
            if i == j:
                dist[i][j] = np.inf
    '''
    the double loop is to calculate a dist matrix with femmir model
    1. create a dist of size (query_indices, db_indices)
    2. initialize with inf
    3. populate with calculated prediction (direct dist)
    '''
    for idx1, item1 in enumerate(query_items):
        query_obj_prop = mgid2predicted_properties_dict[item1]
        for idx2, item2 in enumerate(db_items):
            json_data = {}
            test_obj_prop = mgid2predicted_properties_dict[item2]
            json_data['graph_1'] = [[0, 1], [1, 2], [1, 3], [1, 4]]
            json_data['graph_2'] = [[0, 1], [1, 2], [1, 3], [1, 4]]
            json_data['labels_1'] = [50, 51, query_obj_prop[1], query_obj_prop[2],
                                     query_obj_prop[3]]  # 50: root, 51: person
            json_data['labels_2'] = [50, 51, test_obj_prop[1], test_obj_prop[2], test_obj_prop[3]]
            json_data['ged'] = trunc_train_dist[idx1][idx2]
            # json_data['gt_ged'] = trunc_train_gt_dist[idx1][idx2]
            # json_data['ged'] = trunc_train_gt_dist[idx1][idx2]          # especially just for backward comp. and target
                                                                          # compare assigning the gt to ged for test

            data = trainer.transfer_to_torch(json_data)
            # both target and prediction comes out as probability
            # target = data["target"][0].item()                   # data["target"]

            prediction = trainer.model(data)[0][0].item()       # trainer.model(data)

            ## just taking the real GED to test
            # prediction = trunc_train_gt_dist[idx1][idx2]        # this should output MAP:1 for text-text
            # prediction = np.exp((-1)*(trunc_train_gt_dist[idx1][idx2] / (0.5 * (len(json_data["labels_1"]) + len(json_data["labels_2"])))))
            ##
            # convert back to ged from prob
            predicted_ced = np.log(prediction)
            predicted_ced = (-1) * predicted_ced * 0.5
            predicted_ced = predicted_ced * (len(json_data["labels_1"]) + len(json_data["labels_2"]))
            # predicted_ced = prediction

            # print(predicted_ced)
            # print(trunc_train_gt_dist[idx1][idx2])

            dist[idx1][idx2] = predicted_ced

    # print(dist)
    # print(trunc_train_gt_dist)
    fx_calc_map_label(db_indices, query_indices, dist, trunc_train_gt_dist)

    # implement the MAP calc. used for EARS
    # start with sorting; ord = trunc_dist.argsort(1)  # Returns the indices that would sort this array.

if __name__ == "__main__":
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    """
    load all the data
    """
    with open("./data/testset/texttestset.pkl", "rb") as f:
        texttestset = pickle.load(f)
    with open("./data/testset/imagetestset.pkl", "rb") as f:
        imagetestset = pickle.load(f)
    with open("./data/testset/videotestset.pkl", "rb") as f:
        videotestset = pickle.load(f)
    with open("../mgid2predicted_prop_dict.pkl", "rb") as f:
        mgid2predicted_properties_dict = pickle.load(f)
    with open("../testsetitem2idx.pkl", "rb") as f:
        item2idx = pickle.load(f)
    with open("../testsetidx2item.pkl", "rb") as f:
        idx2item = pickle.load(f)
    with open("../predicted_test_distance_matrix.pkl", "rb") as f:
        predicted_test_dist = pickle.load(f)
    with open("../test_distance_matrix.pkl", "rb") as f:
        ground_test_dist = pickle.load(f)
    with open("../testset.pkl", "rb") as f:
        testset = pickle.load(f)

    trainer.model.load_state_dict(torch.load(args.load_path))
    trainer.model.eval()
    # main()
    maps = []
    maps.append(calc_ced(texttestset, texttestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict, trainer))
    maps.append(calc_ced(videotestset, texttestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict, trainer))
    maps.append(calc_ced(imagetestset, texttestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict, trainer))
    maps.append(calc_ced(testset, texttestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict, trainer))
    #
    maps.append(calc_ced(videotestset, videotestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict,
             trainer))
    maps.append(calc_ced(texttestset, videotestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict,
             trainer))
    maps.append(calc_ced(imagetestset, videotestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict,
             trainer))
    maps.append(calc_ced(testset, videotestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict,
             trainer))
    #
    maps.append(calc_ced(imagetestset, imagetestset, predicted_test_dist, ground_test_dist, item2idx,
             mgid2predicted_properties_dict,
             trainer))
    maps.append(calc_ced(texttestset, imagetestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict,
             trainer))
    maps.append(calc_ced(videotestset, imagetestset, predicted_test_dist, ground_test_dist, item2idx,
             mgid2predicted_properties_dict,
             trainer))
    maps.append(calc_ced(testset, imagetestset, predicted_test_dist, ground_test_dist, item2idx, mgid2predicted_properties_dict,
             trainer))

    print(np.mean(maps))
