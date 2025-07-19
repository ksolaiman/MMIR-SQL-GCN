"""SimGNN class and runner."""

import glob
import math
import random
import sys

import numpy as np
import torch
from torch_geometric.nn import GCNConv
from tqdm import tqdm, trange
import pickle
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from param_parser import parameter_parser

from layers import AttentionModule, TenorNetworkModule
# from utils import process_pair, calculate_loss, calculate_normalized_ged

from create_femmir_pairs import create_pairs_for_full_training_pool
from itertools import chain
from create_harg import load_simgnn_graph_json_by_id
from utils import tab_printer

# ----------------------------------------------------------
# Dataset of <query, target> PAIRS
# created from a list of item-ids
# ----------------------------------------------------------
class FemmirDataset(data.Dataset):
    def __init__(self, item_ids, noisy_global_labels=None, split="train"):
        # image_dir, imagetestset, input_transform = None, mgid2predicted_properties_dict=None, label=None, type='image'
        super(FemmirDataset, self).__init__()

        graph_parent_dir = os.path.join(config["dataset_dir"], config.get("simgnn_graph_dir"), "noisy")
        with open(os.path.join(graph_parent_dir, "label_vocab.json"), "r") as f:
            noisy_global_labels = json.load(f)

        self.split = split
        self.global_labels = noisy_global_labels
        self.number_of_labels = len(self.global_labels)

        # self.item_ids = item_ids                          # omitting it as dont feel like would need to access in future
        # self.pairs = read the pairs from the item_ids     # NO
        # just call the methods from create_femmir_pairs in runtime, anyway they would be in cache memory 
        # during training
        if config["create_new_pairs"]:
            # Load distance matrix and mappings
            with open(os.path.join(config["dataset_dir"], "dist_matrices", split if split=="test" else "train", "ground_distance_matrix.pkl"), "rb") as f:
                distance_matrix = pickle.load(f)

            with open(os.path.join(config["dataset_dir"], "dist_matrices", split if split=="test" else "train", "item2idx.pkl"), "rb") as f:
                item2idx = pickle.load(f)

            with open(os.path.join(config["dataset_dir"], "dist_matrices", split if split=="test" else "train", "item2modality.pkl"), "rb") as f:
                item2modality = pickle.load(f)
            positive_pairs, negative_pairs = create_pairs_for_full_training_pool(item_ids, 
                                                                                 item2idx, 
                                                                                 item2modality, 
                                                                                 distance_matrix, 
                                                                                 NOS_mod=5, 
                                                                                 save_pairs=True, 
                                                                                 split=self.split, 
                                                                                 RANDOM_SEED=config["random_seed"]
                                                                                 )  # 5 pos and 5 neg pairs, save=False, seed=42
        else:
            pair_dir = os.path.join(config["dataset_dir"], config.get("pair_save_dir"), self.split)
            positive_pairs = utils.load_pickle(os.path.join(pair_dir, f"positive_pairs_by_queryid.pkl"))
            negative_pairs = utils.load_pickle(os.path.join(pair_dir, f"negative_pairs_by_queryid.pkl"))

        self.pairs = list(chain.from_iterable(positive_pairs.values())) \
                                                + list(chain.from_iterable(negative_pairs.values()))
        
        # DEBUG
        print(f"DEBUG: {self.pairs[0]}")    # each self.pairs item is a tuple (query-id, target-id, ced, (modq, modc))
        print(f"DEBUG: {type(self.pairs), len(self.pairs)}")
        print(f"DEBUG: {self.pairs[:5]}")  # preview first 5

    def __getitem__(self, index):

        # ced was generated from "gold" properties and used as label in simgnn
        query_id, target_id, ced, pairs_modality = self.pairs[index]

        # use graphs created from "noisy" properties for training and testing
        query_data = load_simgnn_graph_json_by_id(config, query_id, self.split if self.split != "valid" else "train", noisy=True)
        target_data = load_simgnn_graph_json_by_id(config, target_id, self.split if self.split != "valid" else "train", noisy=True)

        edges_1 = query_data["graph"] # + [[y, x] for x, y in query_data["graph"]]
        edges_2 = target_data["graph"] # + [[y, x] for x, y in target_data["graph"]]

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        # this are single data, not batch, so 2D; Transpose should happen when u read batch data in .fit()

        #####
        # edges_1 = torch.Tensor(data["graph_1"]).type(torch.long)
        # edges_2 = torch.Tensor(data["graph_1"]).type(torch.long)
        ###
        
        assert max(query_data["labels"] + target_data["labels"]) < len(self.global_labels)

        # the next two lines are redundant now, as i already map to global_labels during creating simgnn graph
        # features_1 = torch.Tensor([self.global_labels[i] for i in query_data["labels"]]).long()
        # features_2 = torch.Tensor([self.global_labels[i] for i in target_data["labels"]]).long()

        features_1 = torch.tensor(query_data["labels"]).long()
        features_2 = torch.tensor(target_data["labels"]).long()

        features_1 = F.one_hot(features_1, num_classes=len(self.global_labels)).float()
        features_2 = F.one_hot(features_2, num_classes=len(self.global_labels)).float()

        # create similarity score
        norm_ged = ced / (0.5 * (len(query_data["labels"]) + len(target_data["labels"])))
        target = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()

        return edges_1, edges_2, features_1, features_2, target #, {self.pairs[index]}

        # return self.pairs[index]

    def __len__(self):
        return len(self.pairs)    # each self.pairs item is a tuple (query-id, target-id, ced, (modq, modc))


class SimGNN(torch.nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        self.setup_layers()

    # basically number of input features for FC layer
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.number_labels, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        # self.convolution_3 = GCNConv(self.args.filters_1, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features

    def forward(self, edge_index_1, edge_index_2, features_1, features_2):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        # edge_index_1 = data["edge_index_1"]
        # edge_index_2 = data["edge_index_2"]
        # features_1 = data["features_1"]
        # features_2 = data["features_2"]
        
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))

        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = scores.squeeze()   # torch.transpose(scores, 1, 2)  # torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        return score
    

class SimGNNTrainer(object):
    """
    SimGNN model trainer.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """

        self.args = args
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # self.initial_label_enumeration()
        # self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        # self.val_graphs = glob.glob(self.args.validation_graphs + "*.json")
        # with open("selfObj.pkl", "rb") as f:
        #      obj = pickle.load(f)
        # #      self.training_graphs = obj.training_graphs
        # #      self.val_graphs = obj.val_graphs # glob.glob(self.args.validation_graphs + "*.json")
        # #      self.testing_graphs = obj.testing_graphs # glob.glob(self.args.testing_graphs + "*.json")
        #      self.global_labels = obj.global_labels
        #      self.number_of_labels = len(self.global_labels)
        

        # self.training_dataset = FemmirDataset(self.training_graphs, self.global_labels)
        # self.train_dataloader = DataLoader(dataset=self.training_dataset, num_workers=args.workers, batch_size=args.batch_size, \
        #     shuffle=False, drop_last=True) # last batch may be < 128/intended sz, drop it then, to avoid calc error in future

        # self.validation_dataset = FemmirDataset(self.val_graphs, self.global_labels)
        # self.validation_dataloader = DataLoader(dataset=self.validation_dataset, num_workers=args.workers, batch_size=args.batch_size, \
        #     shuffle=False, drop_last=True)

        # Load everything once
        all_data = load_all_splits(config.get('dataset_dir'))
        
        # ------------------------------------------------------------------------------------------------
        # PREPARE data and use the following itemsets to create future <query, target> pairs for SIMGNN
        # pass each one of them to FemmIRDataset Class now to create the pairs
        # ------------------------------------------------------------------------------------------------
        trainids, validids, testids = prepare_full_valid_and_undersample_balanced_train_data(all_data, random_seed=config["random_seed"])  # 80-20-20 split

        self.trainset = FemmirDataset(trainids, split="train")
        self.validset = FemmirDataset(validids, split="valid")
        self.testset = FemmirDataset(testids, split="test")

        self.global_labels = self.trainset.global_labels            # already have <UNK> to deal with unknown node-labels in valid and test
        self.number_of_labels = len(self.global_labels)

        print(self.trainset[0])

        # shuffle the data to uncluster to same item-id
        self.train_dataloader = DataLoader(dataset=self.trainset, num_workers=args.workers, batch_size=args.batch_size, \
                                           shuffle=True, drop_last=True)
        self.validation_dataloader = DataLoader(dataset=self.validset, num_workers=args.workers, batch_size=args.batch_size, \
                                           shuffle=True, drop_last=True)

        self.setup_model()
        
    def setup_model(self):
        """
        Creating a SimGNN.
        """
        self.model = SimGNN(self.args, self.number_of_labels).to(self.device)

    # basically creating node-label mapping, so color mapping na krleo hoto :(
    # takes a long time for full data, can reuse
    def initial_label_enumeration(self):
        """
        Collecting the unique node idsentifiers.
        """
        print("\nEnumerating unique labels.\n")
        self.training_graphs = glob.glob(self.args.training_graphs + "*.json")
        self.val_graphs = glob.glob(self.args.validation_graphs + "*.json")
        self.testing_graphs = glob.glob(self.args.testing_graphs + "*.json")
        graph_pairs = self.training_graphs + self.testing_graphs + self.val_graphs
        self.global_labels = set()
        for graph_pair in tqdm(graph_pairs):
            data = process_pair(graph_pair)
            self.global_labels = self.global_labels.union(set(data["labels_1"]))
            self.global_labels = self.global_labels.union(set(data["labels_2"]))
        self.global_labels = list(self.global_labels)
        self.global_labels = {val: index for index, val in enumerate(self.global_labels)}
        self.number_of_labels = len(self.global_labels)

        print(self.number_of_labels)

    def create_batches(self):
        """
        Creating batches from the training graph list.
        :return batches: List of lists with batches.
        """
        random.seed(42)
        random.shuffle(self.training_graphs)
        batches = []
        for graph in range(0, len(self.training_graphs), self.args.batch_size):
            batches.append(self.training_graphs[graph:graph+self.args.batch_size])
        return batches

    def transfer_to_torch(self, data):
        """
        Transferring the data to torch and creating a hash table.
        Including the indices, features and target.
        :param data: Data dictionary.
        :return new_data: Dictionary of Torch Tensors.
        """
        # new_data = dict()
        edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]

        edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

        features_1, features_2 = [], []
        features_1 = [[1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()] for n in data["labels_1"]]
        features_2 = [[1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()] for n in data["labels_2"]]

        # for n in data["labels_1"]:
        #     features_1.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        # for n in data["labels_2"]:
        #     features_2.append([1.0 if self.global_labels[n] == i else 0.0 for i in self.global_labels.values()])

        # if "gt_ged" in data:
        #     norm_gt_ged = data["gt_ged"]/(0.5*(len(data["labels_1"])+len(data["labels_2"])))
        #     new_data["target_gt"] = torch.from_numpy(np.exp(-norm_gt_ged).reshape(1, 1)).view(-1).float()

        norm_ged = data["ged"] / (0.5 * (len(data["labels_1"]) + len(data["labels_2"])))
        target = torch.from_numpy(np.exp(-norm_ged).reshape(1, 1)).view(-1).float()
        return edges_1, edges_2, features_1, features_2, target

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        
        edges_1 = list()
        edges_2 = list()
        features_1 = list()
        features_2 = list()
        target = torch.empty((len(batch), 1), dtype=torch.float)
        for idx, graph_pair in enumerate(batch):
            data = process_pair(graph_pair)
            temp_edges_1, temp_edges_2, temp_features_1, \
                temp_features_2, temp_target = self.transfer_to_torch(data)
            edges_1.append(temp_edges_1)
            edges_2.append(temp_edges_2)
            features_1.append(temp_features_1)
            features_2.append(temp_features_2)
            target[idx] = temp_target
        
        target = target.to(self.device)

        edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
        edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

        features_1 = torch.FloatTensor(np.array(features_1))
        features_2 = torch.FloatTensor(np.array(features_2))
            
        # print(features_1.shape)       # torch.Size([128, 5, 12]) 
        # print(edges_1.shape)          # torch.Size([2, 8, 128]) 

        edge_index_1 = edges_1.to(self.device)
        edge_index_2 = edges_2.to(self.device)

        features_1 = features_1.to(self.device)
        features_2 = features_2.to(self.device)
        # print(data)
        # input("wait")
        # print(target)

        ## can't do self.model(data)[0] anymore, that gets just first sample
        prediction = self.model(edge_index_1, edge_index_2, features_1, features_2)#[0]
        losses = torch.nn.functional.mse_loss(target, prediction).sum()

        losses.backward() # losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def gpu_process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        
        (edges_1, edges_2, features_1, features_2, target) = batch

        target = target.to(self.device)

        # THe reshaping of edges
        edges_1 = edges_1.reshape(2, -1, edges_1.size(0)) # first index is bvatch_sz, 128++, (128, 2, 8)
        edges_2 = edges_2.reshape(2, -1,  edges_2.size(0)) # torch.transpose(edges_2, 0, 2)
            
        # print(features_1.shape)       # torch.Size([128, 5, 12]) 
        # print(edges_1.shape)          # torch.Size([2, 8, 128]) 


        edge_index_1 = edges_1.to(self.device)
        edge_index_2 = edges_2.to(self.device)

        features_1 = features_1.to(self.device)
        features_2 = features_2.to(self.device)
        # print(data)
        # input("wait")
        # print(target)

        ## can't do self.model(data)[0] anymore, that gets just first sample
        prediction = self.model(edge_index_1, edge_index_2, features_1, features_2) #[0]
        losses = torch.nn.functional.mse_loss(target, prediction).mean()   #.sum()

        losses.backward() # losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Fitting a model.
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)

        self.model.train()
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")
        
        best_val_loss = float("inf")
        best_model = None
        stale = 0
        
        for epoch in epochs:
            # batches = self.create_batches()
            self.loss_sum = 0
            main_index = 0

            # for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            #     loss_score = self.process_batch(batch)
            for index, batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Batches"):
                loss_score = self.gpu_process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))
                
            
            # run model on validation data to see if more epochs are needed or not
            self.model.eval()
            val_loss = 0
            size = len(self.validation_dataloader.dataset)
            num_batches = len(self.validation_dataloader)

            with torch.no_grad():
                for X1, X2, X3, X4, y in tqdm(self.validation_dataloader):
                    y = y.to(self.device)

                    # THe reshaping of edges
                    X1 = X1.reshape(2, -1, X1.size(0)) # first index is bvatch_sz, 128++, (128, 2, 8)
                    X2 = X2.reshape(2, -1,  X2.size(0)) # torch.transpose(edges_2, 0, 2)

                    X1 = X1.to(self.device)
                    X2 = X2.to(self.device)

                    X3 = X3.to(self.device)
                    X4 = X4.to(self.device)
                    prediction = self.model(X1, X2, X3, X4)
                    val_loss += torch.nn.functional.mse_loss(y, prediction).item()

                # val_loss = losses.item()/len(self.val_graphs)
                val_loss /= num_batches
                print(f"Validation Avg loss: {val_loss:>8f} \n")


                if val_loss < best_val_loss:
                    stale = 0
                    best_val_loss = val_loss
                    best_model = self.model
                else:
                    stale += 1
                    if stale > 20:
                        break
                        
                        
        from datetime import datetime
        now = datetime.now()
        dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        # with open('./epoch_'+str(epoch)+'_'+'_'+str(best_val_loss)+'_'+dt_string+'.pt', "wb+") as mf:
        #     torch.save(best_model.state_dict(), mf)
        with open(self.args.save_path +'epoch_'+str(epoch)+'_'+'_'+str(best_val_loss)+'_'+dt_string+'.pt', "wb+") as mf:
            torch.save(best_model.state_dict(), mf)
            # torch.save(best_model, mf)
        self.model = best_model


    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        # model = TheModelClass(*args, **kwargs)
        # print(self.model)
        self.model.load_state_dict(torch.load(self.args.load_path))
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        
        predictions = []
        targets = []
        
        from sklearn.metrics import average_precision_score
        no_of_nodes_in_pairs = list()
        for graph_pair in tqdm(self.testing_graphs):
            data = process_pair(graph_pair)
            self.ground_truth.append(calculate_normalized_ged(data))
            no_of_nodes_in_pairs.append(len(data["labels_1"])+len(data["labels_2"]))
            data = self.transfer_to_torch(data)
            target = data["target"]
            prediction = self.model(data)
            targets.append(target[0].item())
            predictions.append(prediction[0][0].item())
            self.scores.append(calculate_loss(prediction, target))

        self.print_evaluation()

        targets = [1 if math.ceil(-1*math.log(item)*no_of_nodes_in_pairs[idx]) <= 2 else 0 for idx, item in enumerate(targets)] # calculates the number of total relevant items in dataset
        predictions = [1 if math.ceil(-1*math.log(item)*no_of_nodes_in_pairs[idx]) <= 2 else 0 for idx, item in enumerate(predictions)] # calculates the number of total relevant items obtained

        average_precision = average_precision_score(targets, predictions)

        print('Average precision-recall score: {0:0.2f}'.format(
              average_precision))
        
        from sklearn.metrics import precision_recall_fscore_support
        result = precision_recall_fscore_support(targets, predictions, beta=1, average='binary')
        print(result)

    def score_for_map_rough(self):
        """
        Scoring on the test set.
        """
        # print("\n\n Single data Model evaluation.\n")
        self.model.load_state_dict(torch.load(self.args.load_path))
        self.model.eval()
        self.scores = []
        self.ground_truth = []

        predictions = []
        targets = []

        from sklearn.metrics import average_precision_score
        no_of_nodes_in_pairs = list()
        for graph_pair in tqdm(self.testing_graphs):
            # data = process_pair(graph_pair)
            self.ground_truth.append(calculate_normalized_ged(data))
            no_of_nodes_in_pairs.append(len(data["labels_1"]) + len(data["labels_2"]))
            data = self.transfer_to_torch(data)
            target = data["target_gt"]
            prediction = self.model(data)
            targets.append(target[0].item())
            predictions.append(prediction[0][0].item())
            self.scores.append(calculate_loss(prediction, target))

        self.print_evaluation()

        targets = [1 if math.ceil(-1 * math.log(item) * no_of_nodes_in_pairs[idx]) <= 2 else 0 for idx, item in
                   enumerate(targets)]  # calculates the number of total relevant items in dataset
        predictions = [1 if math.ceil(-1 * math.log(item) * no_of_nodes_in_pairs[idx]) <= 2 else 0 for idx, item in
                       enumerate(predictions)]  # calculates the number of total relevant items obtained

        average_precision = average_precision_score(targets, predictions)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        from sklearn.metrics import precision_recall_fscore_support
        result = precision_recall_fscore_support(targets, predictions, beta=1, average='binary')
        print(result)

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        norm_ged_mean = np.mean(self.ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in self.ground_truth])
        model_error = np.mean(self.scores)
        print("\nBaseline error: " +str(round(base_error, 5))+".")
        print("\nModel test error: " +str(round(model_error, 5))+".")



def prepare_full_valid_and_undersample_balanced_train_data(all_data, random_seed=42):

    # Access modality-specific train pools
    image_pool = all_data['image_trainpool']
    video_pool = all_data['video_trainpool']
    text_pool = all_data['text_trainpool']

    # Access overall pools
    train_pool = all_data['trainpool']
    testset = all_data['testset']

    full_train_ids, full_val_ids = train_validation_split(train_pool, val_ratio=0.2, random_seed=random_seed)
    full_val_set = set(full_val_ids)

    # Next: remove validation IDs from subpools
    def remove_val_ids(subpool, val_set, label):
        filtered = [x for x in subpool if x not in val_set]
        removed = [x for x in subpool if x in val_set]
        print(f"{label} subpool: {len(removed)} removed due to validation overlap")
        return filtered

    image_pool = remove_val_ids(image_pool, full_val_set, "Image")
    video_pool = remove_val_ids(video_pool, full_val_set, "Video")
    text_pool = remove_val_ids(text_pool, full_val_set, "Text")


    train_ids, val_ids, details = stratified_train_validation_split_by_modality(image_pool, video_pool, text_pool,
                                                                        balanced=True, undersample=True, 
                                                                        val_ratio=0, random_seed=random_seed)
    

    # print(f"Size of balanced training data: {len(train_ids)}")
    # print(f"Size of full validation data: {len(full_val_ids)}")

    # Convert all to sets for easy comparison
    set_full_val = set(full_val_ids)
    set_val = set(val_ids)
    set_train = set(train_ids)

    # 1. Check all val_ids are inside full_val_ids
    print("val_ids ⊆ full_val_ids:", set_val.issubset(set_full_val))

    # 2. Check train_ids and full_val_ids are disjoint
    print("train_ids ∩ full_val_ids == ∅:", set_train.isdisjoint(set_full_val))

    # (Optional) Print any mismatches
    if not set_val.issubset(set_full_val):
        print("val_ids not in full_val_ids:", set_val - set_full_val)

    if not set_train.isdisjoint(set_full_val):
        print("train_ids that are wrongly in full_val_ids:", set_train & set_full_val)

    return train_ids, full_val_ids, testset


from utils import load_config
from set_train_test_data_and_distance_matrix import *

config = load_config()
DATASET_DIR = config["dataset_dir"]

def usage():
    
    # Load everything once
    all_data = load_all_splits(config.get('dataset_dir'))
    
    # ------------------------------------------------------------------------------------------------
    # PREPARE data and use the following itemsets to create future <query, target> pairs for SIMGNN
    # pass each one of them to FemmIRDataset Class now to create the pairs
    # ------------------------------------------------------------------------------------------------
    trainids, validids, testids = prepare_full_valid_and_undersample_balanced_train_data(all_data, random_seed=config["random_seed"])  # 80-20-20 split

    print(f"Size of balanced training ids: {len(trainids)}")
    print(f"Size of full validation ids: {len(validids)}")
    print(f"Size of full test ids: {len(testids)}")

    trainset = FemmirDataset(trainids, split="train")
    validset = FemmirDataset(validids, split="valid")
    testset = FemmirDataset(testids, split="test")
    print(trainset[0])

    train_loader = DataLoader(dataset=trainset, num_workers=2, batch_size=128, shuffle=False)
    print(train_loader)
    
    # data_set = CubTextDataset(data_dir, test_list, split)
    # data_loader = DataLoader(dataset=data_set, num_workers=args.workers, batch_size=args.batch_size, shuffle=False)

if __name__ == "__main__":
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    if not args.evaluate_only:
        trainer.fit()
    else:
        trainer.score()