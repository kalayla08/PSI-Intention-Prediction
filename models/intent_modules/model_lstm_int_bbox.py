import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, VGG16_Weights

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


class LSTMIntBbox(nn.Module):
    def __init__(self, args, model_configs):
        super(LSTMIntBbox, self).__init__()
        self.args = args
        self.model_configs = model_configs
        self.observe_length = self.args.observe_length
        self.predict_length = self.args.predict_length

        # Initialisation du backbone avec une instance du modèle spécifié
        if self.args.backbone == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif self.args.backbone == 'vgg16':
            self.backbone = models.vgg16(weights=VGG16_Weights.DEFAULT)
        else:
            self.backbone = None
            
        self.intent_predictor = LSTMInt(self.args, self.model_configs['intent_model_opts'])
        self.module_list = self.intent_predictor.module_list
        self.network_list = [self.intent_predictor]

    def forward(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        dec_input_emb = None  # as the additional emb for intent predictor
        assert bbox.shape[1] == self.observe_length

        # 1. backbone feature (to be implemented for images)
        if self.backbone is not None:
            pass  # If required, add the image processing part here

        # 2. intent prediction
        intent_pred = self.intent_predictor(bbox, dec_input_emb)
        return intent_pred.squeeze()

    def build_optimizer(self, args):
        param_group = []
        learning_rate = args.lr
        if self.backbone is not None:
            for name, param in self.backbone.named_parameters():
                if not self.args.freeze_backbone:
                    param.requires_grad = True
                    param_group += [{'params': param, 'lr': learning_rate * 0.1}]
                else:
                    param.requires_grad = False

        for net in self.network_list:
            for module in net.module_list:
                param_group += [{'params': module.parameters(), 'lr': learning_rate}]

        optimizer = torch.optim.Adam(param_group, lr=args.lr, eps=1e-7)
        for param_group in optimizer.param_groups:
            param_group['lr0'] = param_group['lr']

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, scheduler

    def lr_scheduler(self, cur_epoch, args, gamma=10, power=0.75):
        decay = (1 + gamma * cur_epoch / args.epochs) ** (-power)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr0'] * decay
            param_group['weight_decay'] = 1e-3
            param_group['momentum'] = 0.9
            param_group['nesterov'] = True
        return

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)

    def predict_intent(self, data):
        bbox = data['bboxes'][:, :self.args.observe_length, :].type(FloatTensor)
        dec_input_emb = None  # as the additional emb for intent predictor
        assert bbox.shape[1] == self.observe_length

        # 1. backbone feature (to be implemented if needed)
        if self.backbone is not None:
            pass  # If required, add the image processing part here

        # 2. intent prediction
        intent_pred = self.intent_predictor(bbox, dec_input_emb)
        return intent_pred.squeeze()

class LSTMInt(nn.Module):
    def __init__(self, args, model_opts):
        super(LSTMInt, self).__init__()

        enc_in_dim = model_opts['enc_in_dim']
        enc_out_dim = model_opts['enc_out_dim']
        output_dim = model_opts['output_dim']
        n_layers = model_opts['n_layers']
        dropout = model_opts['dropout']

        self.args = args

        self.enc_in_dim = enc_in_dim  # input bbox+convlstm_output context vector
        self.enc_out_dim = enc_out_dim
        self.encoder = nn.LSTM(
            input_size=self.enc_in_dim,
            hidden_size=self.enc_out_dim,
            num_layers=n_layers,
            batch_first=True,
            bias=True
        )

        self.output_dim = output_dim  # 2/3: intention; 62 for reason; 1 for trust score; 4 for trajectory.

        self.fc = nn.Sequential(
            nn.Linear(self.enc_out_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, self.output_dim)
        )

        # Activation is handled separately
        self.module_list = [self.encoder, self.fc]

    def forward(self, enc_input, dec_input_emb=None):
        enc_output, (enc_hc, enc_nc) = self.encoder(enc_input)
        enc_last_output = enc_output[:, -1, :]  # bs x hidden_dim
        output = self.fc(enc_last_output)

        if self.args.intent_num == 2:
            return torch.sigmoid(output)  # For binary classification
        elif self.args.intent_num == 3:
            return torch.softmax(output, dim=1)  # For multi-class classification
        else:
            return output  # No activation

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
