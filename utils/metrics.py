from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate_intent(target, target_prob, prediction, args):
    """
    Evaluate the intent prediction performance.
    
    :param target: (bs x 1), hard label; target_prob: soft probability, 0-1, agreement mean([0, 0.5, 1]).
    :param prediction: (bs), sigmoid probability, 1-dim, should use 0.5 as threshold
    :return: Dictionary with evaluation metrics.
    """
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

    print(f"target_prob shape: {target_prob.shape}")
    print(f"prediction shape: {prediction.shape}")

    if len(target_prob.shape) == 1:
        target_prob = target_prob.reshape(-1, 1)
    if len(prediction.shape) == 1:
        prediction = prediction.reshape(-1, 1)

    if target_prob.shape[1] != prediction.shape[1]:
        print(f"Erreur de dimension: target_prob shape: {target_prob.shape}, prediction shape: {prediction.shape}")
        raise ValueError("Les dimensions de target_prob et prediction ne sont pas cohérentes.")
    
    # Assuming binary classification, take the second column (probability of class 1) if shape is 2
    if target_prob.shape[1] == 2:
        target_prob = target_prob[:, 1]
        prediction = prediction[:, 1]

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
