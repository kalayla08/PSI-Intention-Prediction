from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate_intent(target, target_prob, prediction, args):
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

    target = np.array(target)
    target_prob = np.array(target_prob)
    prediction = np.array(prediction)

    if args.intent_num == 3:
        # Convert target to one-hot encoding if necessary
        if target.ndim == 1:
            target_one_hot = np.zeros((target.size, prediction.shape[1]))
            target_one_hot[np.arange(target.size), target] = 1
        else:
            target_one_hot = target

        lbl_pred = np.argmax(prediction, axis=1)
        lbl_target = np.argmax(target_one_hot, axis=1)
    else:
        lbl_pred = np.round(prediction).astype(int)
        lbl_target = target.astype(int)

    MSE = mean_squared_error(target_prob, prediction)
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

    return results


def shannon(data):
    """
    Calculate Shannon entropy.
    
    :param data: numpy array
    :return: Shannon entropy
    """
    return -np.sum(data * np.log2(data))
