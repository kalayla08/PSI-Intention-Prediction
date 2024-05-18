import os
import numpy as np
from utils.utils import AverageMeter
from utils.metrics import evaluate_intent
import json


class RecordResults():
    def __init__(self, args=None, intent=True, traj=True, reason=False, evidential=False,
                 extract_prediction=False):
        self.args = args
        self.save_output = extract_prediction
        self.intent = intent
        self.traj = traj
        self.reason = reason
        self.evidential = evidential

        self.all_train_results = {}
        self.all_eval_results = {}
        self.all_val_results = {}

        self.result_path = os.path.join(self.args.checkpoint_path, 'results')
        if not os.path.isdir(self.args.checkpoint_path):
            os.makedirs(self.args.checkpoint_path)

        self._log_file = os.path.join(self.args.checkpoint_path, 'log.txt')
        open(self._log_file, 'w').close()

    def log_args(self, args):
        args_file = os.path.join(self.args.checkpoint_path, 'args.txt')
        with open(args_file, 'a') as f:
            json.dump(args.__dict__, f, indent=2)

    def train_epoch_reset(self, epoch, nitern):
        self.log_loss_total = AverageMeter()
        self.log_loss_intent = AverageMeter()
        self.log_loss_traj = AverageMeter()
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        self.traj_gt = []
        self.traj_pred = []
        self.train_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

    def train_intent_batch_update(self, itern, data, intent_gt, intent_prob_gt, intent_prob, loss, loss_intent):
        bs = intent_gt.shape[0]
        self.log_loss_total.update(loss, bs)
        self.log_loss_intent.update(loss_intent, bs)
        if intent_prob != []:
            self.intention_gt.extend(intent_gt)
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)
        if (itern + 1) % self.args.print_freq == 0:
            with open(self.args.checkpoint_path + "/training_info.txt", 'a') as f:
                f.write('Epoch {}/{} Batch: {}/{} | Total Loss: {:.4f} |  Intent Loss: {:.4f} \n'.format(
                    self.epoch, self.args.epochs, itern, self.nitern, self.log_loss_total.avg,
                    self.log_loss_intent.avg))

                
                
                
    def train_intent_epoch_calculate(self, writer):
        print('----------- Training results: ------------------------------------ ')
        if self.intention_pred:
            # Vérifier les dimensions et le type des données
            print(f"intention_gt type: {type(self.intention_gt)}, shape: {np.array(self.intention_gt).shape}")
            print(f"intention_prob_gt type: {type(self.intention_prob_gt)}, shape: {np.array(self.intention_prob_gt).shape}")
            print(f"intention_pred type: {type(self.intention_pred)}, shape: {np.array(self.intention_pred).shape}")

            # Convert lists to numpy arrays and ensure correct dimensions
            lbl_target = np.array(self.intention_gt).flatten()
            lbl_target_prob = np.array(self.intention_prob_gt).reshape(-1, self.args.intent_num)
            lbl_pred = np.array(self.intention_pred).reshape(-1, self.args.intent_num)

            # Vérifier les formes des tableaux après conversion
            print(f"lbl_target shape: {lbl_target.shape}")
            print(f"lbl_target_prob shape: {lbl_target_prob.shape}")
            print(f"lbl_pred shape: {lbl_pred.shape}")

            intent_results = evaluate_intent(lbl_target, lbl_target_prob, lbl_pred, self.args)
            self.train_epoch_results['intent_results'] = intent_results

            if 'ConfusionMatrix' in intent_results:
                # Vérifier les dimensions de la matrice de confusion
                confusion_matrix = intent_results['ConfusionMatrix']
                if len(confusion_matrix) == self.args.intent_num and all(len(row) == self.args.intent_num for row in confusion_matrix):
                    print("La matrice de confusion a les bonnes dimensions.")
                else:
                    print("Erreur: dimensions incorrectes de la matrice de confusion.")
            else:
                print("Erreur: la matrice de confusion n'a pas été calculée.")
        else:
            print("Pas de données d'intention pour calculer les résultats.")

        print('----------------------------------------------------------- ')

        self.all_train_results[str(self.epoch)] = self.train_epoch_results
        self.log_info(epoch=self.epoch, info=self.train_epoch_results, filename='train')

        if writer:
            for key in ['MSE', 'Acc', 'F1', 'mAcc']:
                val = self.train_epoch_results['intent_results'][key]
                writer.add_scalar(f'Train/Results/{key}', val, self.epoch)

            if 'ConfusionMatrix' in self.train_epoch_results['intent_results']:
                confusion_matrix = self.train_epoch_results['intent_results']['ConfusionMatrix']
                for i in range(len(confusion_matrix)):
                    for j in range(len(confusion_matrix[i])):
                        val = confusion_matrix[i][j]
                        writer.add_scalar(f'ConfusionMatrix/train{i}_{j}', val, self.epoch)

    def eval_epoch_reset(self, epoch, nitern, intent=True, traj=True, args=None):
        self.frames_list = []
        self.video_list = []
        self.ped_list = []
        self.intention_gt = []
        self.intention_prob_gt = []
        self.intention_pred = []
        self.intention_rsn_gt = []
        self.intention_rsn_pred = []
        self.traj_gt = []
        self.traj_pred = []
        self.eval_epoch_results = {}
        self.epoch = epoch
        self.nitern = nitern

    def eval_intent_batch_update(self, itern, data, intent_gt, intent_prob, intent_prob_gt, intent_rsn_gt=None, intent_rsn_pred=None):
        bs = intent_gt.shape[0]
        self.frames_list.extend(data['frames'].detach().cpu().numpy())
        assert len(self.frames_list[0]) == self.args.observe_length
        self.video_list.extend(data['video_id'])
        self.ped_list.extend(data['ped_id'])

        if intent_prob != []:
            self.intention_gt.extend(intent_gt)
            self.intention_prob_gt.extend(intent_prob_gt)
            self.intention_pred.extend(intent_prob)
            if intent_rsn_gt is not None:
                self.intention_rsn_gt.extend(intent_rsn_gt)
                self.intention_rsn_pred.extend(intent_rsn_pred)

    def eval_intent_epoch_calculate(self, writer):
        print('----------- Evaluate results: ------------------------------------ ')

        if self.intention_pred:
            intent_results = evaluate_intent(np.array(self.intention_gt), np.array(self.intention_prob_gt),
                                             np.array(self.intention_pred), self.args)
            self.eval_epoch_results['intent_results'] = intent_results
        else:
            print("Pas de données d'intention pour calculer les résultats d'évaluation.")

        print('----------------------finished evalcal------------------------------------- ')
        self.all_eval_results[str(self.epoch)] = self.eval_epoch_results
        self.log_info(epoch=self.epoch, info=self.eval_epoch_results, filename='eval')
        print('log info finished')

        if writer:
            for key in ['MSE', 'Acc', 'F1', 'mAcc']:
                val = self.eval_epoch_results['intent_results'][key]
                writer.add_scalar(f'Eval/Results/{key}', val, self.epoch)

            if 'ConfusionMatrix' in self.eval_epoch_results['intent_results']:
                confusion_matrix = self.eval_epoch_results['intent_results']['ConfusionMatrix']
                for i in range(len(confusion_matrix)):
                    for j in range(len(confusion_matrix[i])):
                        val = confusion_matrix[i][j]
                        writer.add_scalar(f'ConfusionMatrix/eval{i}_{j}', val, self.epoch)

    def log_msg(self, msg: str, filename: str = None):
        if not filename:
            filename = os.path.join(self.args.checkpoint_path, 'log.txt')
        save_to_file = filename
        with open(save_to_file, 'a') as f:
            f.write(str(msg) + '\n')

    def log_info(self, epoch: int, info: dict, filename: str = None):
        if not filename:
            filename = 'log.txt'
        for key in info:
            save_to_file = os.path.join(self.args.checkpoint_path, filename + '_' + key + '.txt')
            self.log_msg(msg='Epoch {} \n --------------------------'.format(epoch), filename=save_to_file)
            with open(save_to_file, 'a') as f:
                if isinstance(info[key], str):
                    f.write(info[key] + "\n")
                elif isinstance(info[key], dict):
                    for k in info[key]:
                        f.write(k + ": " + str(info[key][k]) + "\n")
                else:
                    f.write(str(info[key]) + "\n")
            self.log_msg(msg='.................................................'.format(self.epoch), filename=save_to_file)
