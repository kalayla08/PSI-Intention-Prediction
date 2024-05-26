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
        'ConfusionMatrix': [[]],
    }
    
    bs = target.shape[0]
    lbl_target = target # bs
    lbl_target_prob = target_prob
    
    if args.intent_num == 3:
        lbl_pred = np.argmax(prediction, axis=1) # Convert multilabel-indicator to multiclass
    else: 
        lbl_pred = np.round(prediction)

    MSE = np.mean(np.square(lbl_target_prob - prediction))
    Acc = accuracy_score(lbl_target, lbl_pred)
    F1 = f1_score(lbl_target, lbl_pred, average='macro')
    Precision = precision_score(lbl_target, lbl_pred, average='macro', zero_division=1)
    Recall = recall_score(lbl_target, lbl_pred, average='macro')

    if args.intent_num == 3:
        intent_matrix = confusion_matrix(lbl_target, lbl_pred, labels=[0, 1, 2])
    else:
        intent_matrix = confusion_matrix(lbl_target, lbl_pred)


    results['MSE'] = MSE
    results['Acc'] = Acc
    results['F1'] = F1
    results['Precision'] = Precision
    results['Recall'] = Recall
    results['ConfusionMatrix'] = intent_matrix

    return results

def shannon(data):
    """
    Calculate Shannon entropy.
    
    :param data: numpy array
    :return: Shannon entropy
    """
    return -np.sum(data * np.log2(data))
