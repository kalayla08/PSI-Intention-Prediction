import os
import torch
import json
import numpy as np
from utils.metrics import evaluate_intent

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def validate_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    gt_intent_all = []
    gt_intent_prob_all = []
    pred_intent_all = []

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            intent_pred = model(data)

            if args.intent_num == 2:
                intent_pred = intent_pred.squeeze()  # (bs,)
            elif args.intent_num == 3:
                intent_pred = intent_pred  # (bs, 3)

            gt_intent_all.append(data['intention_binary'][:, args.observe_length].cpu().numpy())
            gt_intent_prob_all.append(data['intention_prob'][:, args.observe_length].cpu().numpy())
            pred_intent_all.append(intent_pred.cpu().numpy())

    gt_intent_all = np.concatenate(gt_intent_all)
    gt_intent_prob_all = np.concatenate(gt_intent_prob_all)
    pred_intent_all = np.concatenate(pred_intent_all)

    print(f"gt_intent_all shape: {gt_intent_all.shape}, pred_intent_all shape: {pred_intent_all.shape}")

    eval_results = evaluate_intent(gt_intent_all, gt_intent_prob_all, pred_intent_all, args)
    print(f"Évaluation des intentions:\n{eval_results}")

    writer.add_scalar('Validation/MSE', eval_results['MSE'], epoch)
    writer.add_scalar('Validation/Accuracy', eval_results['Acc'], epoch)
    writer.add_scalar('Validation/F1', eval_results['F1'], epoch)
    writer.add_scalar('Validation/Precision', eval_results['Precision'], epoch)
    writer.add_scalar('Validation/Recall', eval_results['Recall'], epoch)
    writer.add_scalar('Validation/MeanAccuracy', eval_results['mAcc'], epoch)
    
    recorder.record(epoch, eval_results)

    return recorder


def test_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    gt_intent_all, pred_intent_all, gt_intent_prob_all = [], [], []

    recorder.eval_epoch_reset(epoch, niters)
    for itern, data in enumerate(dataloader):
        intent_logit = model(data)
        if args.intent_num == 3:
            intent_prob = torch.softmax(intent_logit, dim=1).detach().cpu().numpy()
        else:
            intent_prob = torch.sigmoid(intent_logit).detach().cpu().numpy()

        gt_intent = data['intention_binary'][:, args.observe_length].cpu().numpy()
        gt_intent_prob = data['intention_prob'][:, args.observe_length].cpu().numpy()

        gt_intent_all.append(gt_intent)
        gt_intent_prob_all.append(gt_intent_prob)
        pred_intent_all.append(intent_prob)

        recorder.eval_intent_batch_update(itern, data, gt_intent,
                                          intent_prob, gt_intent_prob)

    recorder.eval_intent_epoch_calculate(writer)

    # Concaténer toutes les prédictions et étiquettes
    gt_intent_all = np.concatenate(gt_intent_all)
    gt_intent_prob_all = np.concatenate(gt_intent_prob_all)
    pred_intent_all = np.concatenate(pred_intent_all)

    # Évaluation
    eval_results = evaluate_intent(gt_intent_all, gt_intent_prob_all, pred_intent_all, args)
    print(f"Évaluation des intentions:\n{eval_results}")

    return recorder

def predict_intent(model, dataloader, args, dset='test'):
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        
        # Utiliser la bonne fonction d'activation en fonction du nombre de classes
        if args.intent_num == 2:
            intent_prob = torch.sigmoid(intent_logit).detach().cpu().numpy()
        elif args.intent_num == 3:
            intent_prob = torch.softmax(intent_logit, dim=1).detach().cpu().numpy()
        
        for i in range(len(data['frames'])):
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent
            gt_int = data['intention_binary'][i][args.observe_length].item()  # int list, bs x 60
            gt_int_prob = data['intention_prob'][i][args.observe_length].item()  # float list, bs x 60
            gt_disgr = data['disagree_score'][i][args.observe_length].item()  # float list, bs x 60

            if args.intent_num == 2:
                int_prob = intent_prob[i]
                int_pred = round(int_prob)
            elif args.intent_num == 3:
                int_prob = intent_prob[i]
                int_pred = np.argmax(int_prob)

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['intent'] = int_pred
            dt[vid][pid][fid]['intent_prob'] = int_prob.tolist()  # Convertir en liste pour la sérialisation JSON

    with open(os.path.join(args.checkpoint_path, 'results', f'{dset}_intent_pred.json'), 'w') as f:
        json.dump(dt, f)
