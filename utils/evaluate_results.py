import json
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tabulate import tabulate

def evaluate_intent(groundtruth='', prediction='', args=None):
    with open(groundtruth, 'r') as f:
        gt_intent = json.load(f)

    with open(prediction, 'r') as f:
        pred_intent = json.load(f)

    gt = []
    pred = []
    for vid in gt_intent.keys():
        for pid in gt_intent[vid].keys():
            for fid in gt_intent[vid][pid].keys():
                gt.append(gt_intent[vid][pid][fid]['intent'])
                pred.append(pred_intent[vid][pid][fid]['intent'])
    gt = np.array(gt)
    pred = np.array(pred)
    res = measure_intent_prediction(gt, pred, args)
    print('MSE: ',res['MSE'])
    print('Acc: ', res['Acc'])
    print('F1: ', res['F1'])
    print('Precision: ', res['Precision'])
    print('Recall: ', res['Recall'])
    print('mAcc: ', res['mAcc'])
    print('ConfusionMatrix: ')
    print_confusion_matrix(res['ConfusionMatrix'], args)
    return res['F1'], res['Precision'], res['Recall']



def measure_intent_prediction(target, prediction, args):
    print("Evaluating Intent ...")
    results = {
        #'MSE': 0,
        'Acc': 0,
        'F1': 0,
        'Precision': 0,
        'Recall': 0,
        'mAcc': 0,
        'ConfusionMatrix': [[]],
    }

    bs = target.shape[0]
    lbl_target = np.argmax(target, axis=-1) # bs x ts
    lbl_target = target # bs
    lbl_target_prob = target_prob
    lbl_pred = np.round(prediction) # bs, use 0.5 as threshold

    

    MSE = np.mean(np.square(lbl_target_prob - prediction))

    # Hard label evaluation - acc, f1, precision, recall
    Acc = accuracy_score(lbl_target, lbl_pred) # calculate acc for all samples
    F1_score = f1_score(lbl_target, lbl_pred, average='macro')
    Precision = precision_score(lbl_target, lbl_pred, average='macro', zero_division=1)
    Recall = recall_score(lbl_target, lbl_pred, average='macro')

    if args.intent_num == 3:
        intent_matrix = confusion_matrix(lbl_target, lbl_pred, labels=[0, 1, 2])  # [3 x 3]
    else:
        intent_matrix = confusion_matrix(lbl_target, lbl_pred)  # [2 x 2]


    
    intent_cls_acc = np.array(intent_matrix.diagonal() / intent_matrix.sum(axis=-1)) # 2
    intent_cls_mean_acc = intent_cls_acc.mean(axis=0)
    
    results['MSE'] = MSE
    results['Acc'] = Acc
    results['F1'] = F1_score
    results['Precision'] = Precision
    results['Recall'] = Recall
    results['mAcc'] = intent_cls_mean_acc
    results['ConfusionMatrix'] = intent_matrix
    return results

def print_confusion_matrix(conf_matrix, args):
    class_names = None
    if args.intent_num == 2:
        class_names = ['not_cross', 'cross']
    elif args.intent_num == 3:
        class_names = ['not_cross', 'not_sure', 'cross']

    headers = [""] + class_names
    table = []
    for i, row in enumerate(conf_matrix):
        table.append([class_names[i]] + row.tolist())

    print(tabulate(table, headers=headers, tablefmt="grid"))

