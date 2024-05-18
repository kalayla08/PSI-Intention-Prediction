from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid
import torch.nn.functional as F


def evaluate_intent(target, target_prob, prediction, args):
    '''
    Here we only predict one 'intention' for one track (15 frame observation). (not a sequence as before)
    :param target: (bs x 1), hard label; target_prob: soft probability, 0-1, agreement mean([0, 0.5, 1]).
    :param prediction: (bs), sigmoid probability, 1-dim, should use 0.5 as threshold
    :return:
    '''
    print("Evaluating Intent ...")
    results = {
        'MSE': 0,
        'Acc': 0,
        'F1': 0,
        'Precision': 0,
        'Recall': 0,
        'mAcc': 0,
        'ConfusionMatrix': [[]],
    }

    # Convert inputs to numpy arrays if they are not already
    target = np.array(target)
    target_prob = np.array(target_prob)
    prediction = np.array(prediction)

    # Flatten arrays if they are 2D
    if len(target_prob.shape) == 2 and target_prob.shape[1] > 1:
        # Utiliser la moyenne des probabilités si target_prob contient plusieurs colonnes
        target_prob = np.mean(target_prob, axis=1)
    
    # Vérifier si les dimensions de target_prob et prediction sont cohérentes
    if target_prob.shape != prediction.shape:
        raise ValueError("Les dimensions de target_prob et prediction ne sont pas cohérentes.")
    # Convert probabilities to hard labels using 0.5 threshold
    lbl_pred = np.round(prediction).astype(int)  # Use 0.5 as threshold

    lbl_target = target.astype(int)

    # Calculate mean squared error
    MSE = mean_squared_error(target_prob, prediction)

    # Ensure lbl_target and lbl_pred are valid arrays for accuracy_score
    if lbl_target.ndim == 1 and lbl_pred.ndim == 1:
        Acc = accuracy_score(lbl_target, lbl_pred)
        F1 = f1_score(lbl_target, lbl_pred, average='macro')
        Precision = precision_score(lbl_target, lbl_pred, average='macro', zero_division=1)
        Recall = recall_score(lbl_target, lbl_pred, average='macro')

        if args.intent_num == 3:
            intent_matrix = confusion_matrix(lbl_target, lbl_pred, labels=[0, 1, 2])
        else:
            intent_matrix = confusion_matrix(lbl_target, lbl_pred)

        intent_cls_acc = np.array(intent_matrix.diagonal() / intent_matrix.sum(axis=-1))
        intent_cls_mean_acc = intent_cls_acc.mean(axis=0)

        results['MSE'] = MSE
        results['Acc'] = Acc
        results['F1'] = F1
        results['Precision'] = Precision
        results['Recall'] = Recall
        results['mAcc'] = intent_cls_mean_acc
        results['ConfusionMatrix'] = intent_matrix

    else:
        print("Error: lbl_target and lbl_pred must be 1-dimensional arrays of the same length.")

    return results


def shannon(data):
    shannon = -np.sum(data * np.log2(data))
    return shannon
