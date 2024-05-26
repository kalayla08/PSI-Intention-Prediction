import os
import torch
import json
import numpy as np

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

def validate_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        
        if args.intent_num == 2:  # BCEWithLogitsLoss
            intent_prob = torch.sigmoid(intent_logit)
            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)
        elif args.intent_num == 3:  # CrossEntropyLoss
            intent_prob = torch.softmax(intent_logit, dim=1)
            gt_intent = data['intention_binary'][:, args.observe_length].type(LongTensor)
            gt_intent_prob = torch.zeros_like(intent_logit)  # Ajustez si nécessaire

        recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                          intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

        if itern % args.print_freq == 0:
            print(f"Epoch {epoch}/{args.epochs} | Batch {itern}/{niters}")

    recorder.eval_intent_epoch_calculate(writer)

    return recorder

def test_intent(epoch, model, dataloader, args, recorder, writer):
    model.eval()
    niters = len(dataloader)
    recorder.eval_epoch_reset(epoch, niters)
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        
        if args.intent_num == 2:  # BCEWithLogitsLoss
            intent_prob = torch.sigmoid(intent_logit)
            gt_intent = data['intention_binary'][:, args.observe_length].type(FloatTensor)
            gt_intent_prob = data['intention_prob'][:, args.observe_length].type(FloatTensor)
        elif args.intent_num == 3:  # CrossEntropyLoss
            intent_prob = torch.softmax(intent_logit, dim=1)
            gt_intent = data['intention_binary'][:, args.observe_length].type(LongTensor)
            gt_intent_prob = torch.zeros_like(intent_logit)  # Ajustez si nécessaire

        recorder.eval_intent_batch_update(itern, data, gt_intent.detach().cpu().numpy(),
                                          intent_prob.detach().cpu().numpy(), gt_intent_prob.detach().cpu().numpy())

    recorder.eval_intent_epoch_calculate(writer)

    return recorder

def predict_intent(model, dataloader, args, dset='test'):
    model.eval()
    dt = {}
    for itern, data in enumerate(dataloader):
        intent_logit = model.forward(data)
        
        if args.intent_num == 2:  # BCEWithLogitsLoss
            intent_prob = torch.sigmoid(intent_logit)
        elif args.intent_num == 3:  # CrossEntropyLoss
            intent_prob = torch.softmax(intent_logit, dim=1)

        for i in range(len(data['frames'])):
            vid = data['video_id'][i]  # str list, bs x 60
            pid = data['ped_id'][i]  # str list, bs x 60
            fid = (data['frames'][i][-1] + 1).item()  # int list, bs x 15, observe 0~14, predict 15th intent

            if args.intent_num == 2:  # BCEWithLogitsLoss
                int_prob = intent_prob[i].item()
                int_pred = round(int_prob)  # <0.5 --> 0, >=0.5 --> 1.
            elif args.intent_num == 3:  # CrossEntropyLoss
                int_prob = np.argmax(intent_prob[i].detach().numpy()).item()
                #int_prob = np.argmax(intent_prob[i].detach().cpu().numpy())
                int_pred = int_prob

            if vid not in dt:
                dt[vid] = {}
            if pid not in dt[vid]:
                dt[vid][pid] = {}
            if fid not in dt[vid][pid]:
                dt[vid][pid][fid] = {}
            dt[vid][pid][fid]['intent'] = int_pred
            dt[vid][pid][fid]['intent_prob'] = int_prob

    with open(os.path.join(args.checkpoint_path, 'results', f'{dset}_intent_pred.json'), 'w') as f:
        json.dump(dt, f)

    return dt
