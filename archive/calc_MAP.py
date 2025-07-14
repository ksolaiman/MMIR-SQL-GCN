import pickle
import scipy
import numpy as np
from set_train_test_data_from_mmir_ground import prepare_dataset

with open("predicted_test_distance_matrix.pkl", "rb") as f:
    dist = pickle.load(f)
    print(np.shape(dist))
    
#dist = #scipy.spatial.distance.cdist(test, train, metric) # test is query, train is retrieval set, hahahah
#print(dist[1].reshape(914,2).tolist())
ord = dist.argsort(1) # Returns the indices that would sort this array.
#print(dist[1].reshape(914,2).tolist())
print(ord)

with open("testset.pkl", "rb") as f:
    testset = pickle.load(f)
    
with open("testsetitem2idx.pkl", "rb") as f:
    item2idx = pickle.load(f)
with open("testsetidx2item.pkl", "rb") as f:
    idx2item = pickle.load(f)

with open("test_distance_matrix.pkl", "rb") as f:
    ground_dist = pickle.load(f)
    print(np.shape(ground_dist))


# def fx_calc_map_label(train_items, test_items, dist, ground_dist, k):
#     numcases = len(train_items) #train_labels.shape[0] 
#     test_indices = [item2idx[item] for item in test_items]
#     train_indices = [item2idx[item] for item in train_items]
#     dist = dist[np.ix_(test_indices, train_indices)] # 6 query, so 6 rows, 3 content DB, so 3 columns
    
#     trunc_gt = ground_dist[np.ix_(test_indices, train_indices)]
# #     print(dist)
# #     print(trunc_gt)
#     # assert np.array_equal(dist, dist_temp)
    
#     # we will use the test matrix / dist to calculate the order of the returned items, but we will use ground_dist to calc. relevacne
#     ord = dist.argsort(1) # Returns the indices that would sort this array.
    
#     if k == 0:
#         k = numcases
#     if k == -1:
#         ks = [50, numcases]
#     else:
#         ks = [k]

#     def calMAP(_k):
#         _res = []
#         #for i in range(len(test_item)):
#         for i, mappedi in enumerate(test_indices):
#             order = ord[i]
#             # print(order)
#             p = 0.0
#             r = 0.0
#             for j in range(_k):
#                 #if test_label[i] == train_labels[order[j]]:
#                 #train_item_idx = order[j]
#                 if trunc_gt[i][order[j]] < 2: # only upper color change allowed; order[j] has the mapped indices
#                     #print("ad")
#                     r += 1
#                     p += (r / (j + 1))
#                 else:
#                     #print("nad")
#                     continue
#                 #print(r, p)
#             if r > 0:
#                 _res += [p / r]
#             else:
#                 _res += [0]
#                 #continue
#         #print(_res)
#         return np.mean(_res)

#     res = []
#     for k in ks:
#         res.append(calMAP(k))
#     return res

def calPrecision(y1, x1, y2, x2, x):
    if x2 == x1:
        return 0.0;
    result = (y2-y1)*(x-x1)*1.0/(x2-x1) + y1;
    return result

def fx_calc_map_label(train_items, test_items, dist, ground_dist, k):
    numcases = len(train_items) #train_labels.shape[0] 
    test_indices = [item2idx[item] for item in test_items]
    train_indices = [item2idx[item] for item in train_items]
    dist = dist[np.ix_(test_indices, train_indices)] # 6 query, so 6 rows, 3 content DB, so 3 columns
    
    trunc_gt = ground_dist[np.ix_(test_indices, train_indices)]
#     print(dist)
#     print(trunc_gt)
    # assert np.array_equal(dist, dist_temp)
    
    # we will use the test matrix / dist to calculate the order of the returned items, but we will use ground_dist to calc. relevacne
    ord = dist.argsort(1) # Returns the indices that would sort this array.
    
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        AP = np.zeros((len(test_indices),1));
        recall = np.zeros((len(test_indices),len(train_indices)));
        precision = np.zeros((len(test_indices),len(train_indices)));
        
        print(np.shape(precision))
    
        #for i in range(len(test_item)):
        pi = 0
        rj = 0
        non_zero_indices = list()
        for i, mappedi in enumerate(test_indices):
            order = ord[i]
            ######### relCount = 0 ; is r
            relAll = sum([item < 2.0 for item in trunc_gt[i]])
            if relAll == 0:
                print("zero")
                continue
            else:
                non_zero_indices.append(i)
            p = 0.0
            r = 0.0
            rj = 0
            for j in range(_k):
                #if test_label[i] == train_labels[order[j]]:
                #train_item_idx = order[j]
                if trunc_gt[i][order[j]] < 2: # only upper color change allowed; order[j] has the mapped indices
                    #print("ad")
                    r += 1
                    p += (r / (j + 1))
                    
                    AP[i] = AP[i] + (r/(j+1)); # j+1 as our j starts from 0
                    precision[i,j] = (r/(j+1));
                    recall[i,j] = (r/relAll*1.0);
                    if recall[pi,rj] > 1:
                        print('recall > 1!');
                else:
                    precision[i,j] = (r/(j+1));
                    recall[i,j] = (r/relAll*1.0);
                #print(r, p)
                
                rj += 1
            pi += 1
            if r > 0:
                _res += [p / r]
                AP[i] = AP[i]/r;
            else:
                _res += [0]
                AP[i] = 0;
                #continue
            
        #print(_res)
        map_score = np.mean(AP);
        print(map_score)
        print(len(non_zero_indices))
        #non_zero_indices = [item+1 for item in non_zero_indices]
        
#         print(np.shape(recall))
        recall = np.insert(recall, 0, 0, axis=1)
        recall = np.insert(recall, 0, 0, axis=1)
#         print(np.shape(recall))
        precision = np.insert(precision, 0, 1, axis=1)
        precision = np.insert(precision, 0, 1, axis=1)
#         print(np.shape(precision))
        precisionValue = np.zeros((1000, len(non_zero_indices)));
        count = 0;
        for recallValue in np.arange(0.001, 1, 0.001):
            # print(recallValue)
            count = count + 1;
            # 6×7 logical array
            flag = (recall[:,1:] < recallValue); # don't compare the first column, that's just filler
#             print(np.shape(flag))
#             print(recall[0])
#             print(flag[0])
            flagPlace = np.sum(flag, 1); # just changing axis to 1 from 0 is working miracles!!
#             print(flagPlace)
#             print(np.shape(flagPlace))
            # just got it, why they added 0 and 1 infront of recall and precision, as at 0th threshold, recall is 0, precision is 1 in pr curve, thats why we would do range(1, _k)
            for idx, j in enumerate(non_zero_indices):
                precisionValue[count, idx] = calPrecision(precision[j, flagPlace[j]], recall[j, flagPlace[j]], 
                                                        precision[j, flagPlace[j]+1], recall[j, flagPlace[j]+1],
                                                        recallValue);
                                  
        recallValue = np.transpose([*np.arange(0.001, 1, 0.001)])
        print(recallValue)
        precision = np.mean(precisionValue[1:], 1); # as count starts from 1, first one is always 0, if count start from 1, last one is always 0, instead just start from index 1, then matches with recall calue len
        print(precision)
        #prcurve = [precision,recallValue];
        
        print(np.shape(recallValue))
        print(np.shape(precision))
        
        with open("prcurve/"+str(len(test_indices))+"-"+str(len(train_indices))+"-recall.pkl", "wb") as f:
            pickle.dump(recallValue, f, protocol=2)
        with open("prcurve/"+str(len(test_indices))+"-"+str(len(train_indices))+"-precision.pkl", "wb") as f:
            pickle.dump(precision, f, protocol=2)
#         from matplotlib import pyplot as plt
#         plt.plot(recallValue, precision)
            
        return np.mean(_res)

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res

# dist will change based on what model we test  - SimGNN, SDML or SQL.
# ground_dist will always remain same
# subset of dist is selected in the fx_calc_map_label() function, that is to select subset of items for image/text/video subset test
np.random.seed(1234)
randtest = np.random.choice(testset, 10)
randtrain = np.random.choice(testset, 3)
#print(fx_calc_map_label(testset, testset, dist, ground_dist, k = -1))
#print(fx_calc_map_label(randtrain, randtest, dist, ground_dist, k = 0)) #-1 sets _k to 50
# input()
rows = prepare_dataset()
texttestset = [row[5] for row in rows if row[5]]
imagetestset = [row[3] for row in rows if row[3]]
videotestset = [row[4] for row in rows if row[4]]
print(len(texttestset))
print(len(imagetestset))
print(len(videotestset))

texttrainset = [row[8] for row in rows if row[8]]
imagetrainset = [row[6] for row in rows if row[6]]
videotrainset = [row[7] for row in rows if row[7]]
print(len(texttrainset))
print(len(imagetrainset))
print(len(videotrainset))
textvalidset = [row[11] for row in rows if row[11]]
imagevalidset = [row[9] for row in rows if row[9]]
videovalidset = [row[10] for row in rows if row[10]]
print(len(textvalidset))
print(len(imagevalidset))
print(len(videovalidset))


print(fx_calc_map_label(texttestset, texttestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(videotestset, texttestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(imagetestset, texttestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(testset, texttestset, dist, ground_dist, k = 0))

print(fx_calc_map_label(videotestset, videotestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(texttestset, videotestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(imagetestset, videotestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(testset, videotestset, dist, ground_dist, k = 0))

print(fx_calc_map_label(imagetestset, imagetestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(texttestset, imagetestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(videotestset, imagetestset, dist, ground_dist, k = 0))
print(fx_calc_map_label(testset, imagetestset, dist, ground_dist, k = 0))