import time
import torch
import random
from datapro import CVEdgeDataset
from model import MHMDA, EmbeddingM, EmbeddingD
import numpy as np
from sklearn import metrics
import torch.utils.data.dataloader as DataLoader
from sklearn.model_selection import KFold
import os
import pandas as pd


def save_predictions_labels(test_score, test_label, save_path):
    results = np.vstack((test_label, test_score))
    results_df = pd.DataFrame(results.T, columns=["Labels", "Predictions"])
    results_df.to_csv(save_path, index=False)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def construct_het_mat(rna_dis_mat, dis_mat, rna_mat):  ##*
    mat1 = np.hstack((rna_mat, rna_dis_mat))
    mat2 = np.hstack((rna_dis_mat.T, dis_mat))
    ret = np.vstack((mat1, mat2))
    return ret


def get_metrics(score, label):
    y_pre = score
    y_true = label
    metric = caculate_metrics(y_pre, y_true)
    return metric


def caculate_metrics(pre_score, real_score):
    y_true = real_score
    y_pre = pre_score
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pre, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    precision_u, recall_u, thresholds_u = metrics.precision_recall_curve(y_true, y_pre)
    aupr = metrics.auc(recall_u, precision_u)
    y_score = [0 if j < 0.5 else 1 for j in y_pre]

    acc = metrics.accuracy_score(y_true, y_score)
    f1 = metrics.f1_score(y_true, y_score)
    recall = metrics.recall_score(y_true, y_score)
    precision = metrics.precision_score(y_true, y_score)

    metric_result = [auc, aupr, acc, f1, recall, precision]
    print("One epoch metric： ")
    print_met(metric_result)
    return metric_result


def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'f1_score ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'precision ：%.4f \n' % (list[5]))


def check_input_data(data):
    for i, value in enumerate(data.flatten()):
        if not (0 <= value <= 1):
            print(f"Problematic tensor: {data}")
            print("Shape of the tensor:", data.shape)
            assert 0 <= value <= 1, f"Input data value out of range [0, 1] at index {i}: {value}"


def train_test(simData, train_data, param, state, output_folder):
    epo_metric = []
    valid_metric = []
    all_metrics = []

    train_edges = train_data['train_Edges']
    train_labels = train_data['train_Labels']
    test_edges = train_data['test_Edges']
    test_labels = train_data['test_Labels']

    kfolds = param.kfold
    torch.manual_seed(42)

    if state == 'valid':
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)
        train_idx, valid_idx = [], []
        for train_index, valid_index in kf.split(train_edges):
            train_idx.append(train_index)
            valid_idx.append(valid_index)
        for i in range(kfolds):
            a = i + 1
            model = MHMDA(param, EmbeddingM(param), EmbeddingD(param))
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)

            print(f'################Fold {i + 1} of {kfolds}################')
            edges_train, edges_valid = train_edges[train_idx[i]], train_edges[valid_idx[i]]
            labels_train, labels_valid = train_labels[train_idx[i]], train_labels[valid_idx[i]]
            trainEdges = CVEdgeDataset(edges_train, labels_train)
            validEdges = CVEdgeDataset(edges_valid, labels_valid)
            trainLoader = DataLoader.DataLoader(trainEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)
            validLoader = DataLoader.DataLoader(validEdges, batch_size=param.batchSize, shuffle=True, num_workers=0)

            print("-----training-----")
            for e in range(param.epoch):
                running_loss = 0.0
                epo_label = []
                epo_score = []
                print("epoch：", e + 1)
                model.train()
                start = time.time()
                for i, item in enumerate(trainLoader):
                    data, label = item
                    train_data = data.cuda()
                    true_label = label.cuda()
                    pre_score = model(simData, train_data)
                    train_loss = torch.nn.BCELoss()
                    loss = train_loss(pre_score, true_label)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss += loss.item()
                    print(f"After batch {i + 1}: loss= {loss:.3f};", end='\n')
                    batch_score = pre_score.cpu().detach().numpy()
                    epo_score = np.append(epo_score, batch_score)
                    epo_label = np.append(epo_label, label.numpy())
                end = time.time()
                print('Time：%.2f \n' % (end - start))

            valid_score, valid_label = [], []
            model.eval()
            with torch.no_grad():
                print("-----validing-----")
                for i, item in enumerate(validLoader):
                    data, label = item
                    train_data = data.cuda()
                    pre_score = model(simData, train_data)
                    batch_score = pre_score.cpu().detach().numpy()
                    valid_score = np.append(valid_score, batch_score)
                    valid_label = np.append(valid_label, label.numpy())
                end = time.time()
                print('Time：%.2f \n' % (end - start))

                model_path = os.path.join(output_folder, f"fold_{a}.pkl")
                torch.save(model.state_dict(), model_path)

                metric = get_metrics(valid_score, valid_label)
                all_metrics.append(metric)

            mean_metrics = np.mean(all_metrics, axis=0)

            metrics_path = os.path.join(output_folder, "metrics.txt")
            with open(metrics_path, 'w') as f:
                for metrics in all_metrics:
                    f.write('\t'.join(map(str, metrics)) + '\n')
                f.write("Mean Metrics:\n")
                f.write('\t'.join(map(str, mean_metrics)) + '\n')

    else:
        test_score, test_label = [], []
        testEdges = CVEdgeDataset(test_edges, test_labels)
        testLoader = DataLoader.DataLoader(testEdges, batch_size=param.batchSize, shuffle=False, num_workers=0)
        model = MHMDA(param, EmbeddingM(param), EmbeddingD(param))
        model.load_state_dict(torch.load('./savemodel/fold_1.pkl'))
        model.cuda()
        model.eval()
        with torch.no_grad():
            start = time.time()
            for i, item in enumerate(testLoader):
                data, label = item
                test_data = data.cuda()
                pre_score = model(simData, test_data)
                batch_score = pre_score.cpu().detach().numpy()
                test_score = np.append(test_score, batch_score)
                test_label = np.append(test_label, label.numpy())
            end = time.time()
            print('Time：%.2f \n' % (end - start))
            metrics = get_metrics(test_score, test_label)
        print(np.array(valid_metric))
        cv_metric = np.mean(valid_metric, axis=0)
        print_met(cv_metric)

        return kfolds
