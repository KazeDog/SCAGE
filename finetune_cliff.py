import argparse
import warnings
from pathlib import Path
from typing import Dict, Any, Callable
import os
import sys

import loguru
from tqdm import tqdm

from _config import *


from echo_logger import print_debug, print_info, print_warn, dumps_json
from torch import Tensor, tensor

from data_process.function_group_constant import nfg
from utils.global_var_util import GlobalVar, LossStyle, RoutineControl, REGRESSION_TASK_NAMES, DEFAULTS, \
    set_dist_bar_two, set_dist_bar_three

from utils.userconfig_util import config_current_user, config_dataset_form, get_dataset_form, drop_last_flag
# warnings.filterwarnings("ignore")

import yaml
import shutil
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataloader import CliffDataset
from data_process.data_collator import collator_cliff_pkl
from data_process.compound_tools import CompoundKit
from data_process.split import create_splitter
from models.scage import Scage
from utils.loss_util import bce_loss, get_balanced_atom_fg_loss, use_balanced_atom_fg_loss
from utils.scheduler_util import *
from utils.metric_util import calc_rmse, calc_cliff_rmse
from utils.public_util import set_seed, EarlyStopping

# import F
import torch.nn.functional as F

np.set_printoptions(threshold=10)
torch.set_printoptions(threshold=10)
# torch.set_float32_matmul_precision('high')
# set default device cuda
# torch.set_default_device('cuda')

torch.set_printoptions(sci_mode=False, precision=2, linewidth=400, threshold=1000000000)




# noinspection SpellCheckingInspection
class Trainer(object):
    def __init__(self, config, file_path):
        self.imbalance_ratio = None
        self.config = config

        self.train_loader, self.test_loader = self.get_data_loaders()
        self.net = self._get_net()
        # print_info("Compiling model...")
        # self.net = torch.compile(self.net)
        # print_info("Model compiled!")
        self.criterion = self._get_loss_fn()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()
        loguru.logger.info(f"Optimizer: {self.optim}")
        if config['checkpoint'] and GlobalVar.use_ckpt:
            self.load_ckpt(self.config['checkpoint'])
        else:
            loguru.logger.warning("No checkpoint loaded!")
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_metric = np.inf
            self.writer = SummaryWriter('../train_result/{}/{}_{}_{}_{}_{}'.format(
                'finetune_result_cliff', config['task_name'], config['seed'],
                config['optim']['init_lr'],
                config['batch_size'], datetime.now().strftime('%b%d_%H:%M:%S')
            ))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        copyfile(file_path, self.writer.log_dir)

        self.batch_considered = 200

        self.loss_init = torch.zeros(3, self.batch_considered, device='cuda')
        self.loss_last = torch.zeros(3, self.batch_considered // 10, device='cuda')
        self.loss_last2 = torch.zeros(3, self.batch_considered // 10, device='cuda')
        self.cur_loss_step = torch.zeros(1, dtype=torch.long, device='cuda')
        # self.register_buffer('loss_init', loss_init)
        # self.register_buffer('loss_last', loss_last)
        # self.register_buffer('loss_last2', loss_last2)
        # self.register_buffer('cur_loss_step', cur_loss_step)

    def calc_mt_loss(self, loss_list):
        loss_list = torch.stack(loss_list)

        if self.cur_loss_step == 0:
            self.loss_init[:, 0] = loss_list.detach()
            self.loss_last2[:, 0] = loss_list.detach()
            self.cur_loss_step += 1
            loss_t = (loss_list / self.loss_init[:, 0]).mean()

        elif self.cur_loss_step == 1:

            self.loss_last[:, 0] = loss_list.detach()
            self.loss_init[:, 1] = loss_list.detach()
            self.cur_loss_step += 1
            loss_t = (loss_list / self.loss_init[:, :2].mean(dim=-1)).mean()

        else:
            cur_loss_init = self.loss_init[:, :self.cur_loss_step].mean(dim=-1)
            cur_loss_last = self.loss_last[:, :self.cur_loss_step - 1].mean(dim=-1)
            cur_loss_last2 = self.loss_last2[:, :self.cur_loss_step - 1].mean(dim=-1)
            w = F.softmax(cur_loss_last / cur_loss_last2, dim=-1).detach()
            loss_t = (loss_list / cur_loss_init * w).sum()

            cur_init_idx = self.cur_loss_step.item() % self.batch_considered
            self.loss_init[:, cur_init_idx] = loss_list.detach()

            cur_loss_last2_step = (self.cur_loss_step.item() - 1) % (self.batch_considered // 10)
            self.loss_last2[:, cur_loss_last2_step] = self.loss_last[:, cur_loss_last2_step - 1]
            self.loss_last[:, cur_loss_last2_step] = loss_list.detach()
            self.cur_loss_step += 1
        return loss_t

    def get_data_loaders(self):
        train_dataset = CliffDataset(root=self.config['root'], task_name=self.config['task_name'], mode='train')
        test_dataset = CliffDataset(root=self.config['root'], task_name=self.config['task_name'], mode='test')

        # self.imbalance_ratio = ((dataset.data.label == -1).sum()) / ((dataset.data.label == 1).sum())
        num_workers = self.config['dataloader_num_workers']
        bsz = self.config['batch_size']
        train_loader = DataLoader(train_dataset,
                                  batch_size=bsz,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collator_cliff_pkl,
                                  pin_memory=True,
                                  drop_last=drop_last_flag(len(train_dataset), bsz))
        test_loader = DataLoader(test_dataset,
                                 batch_size=bsz,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 pin_memory=True,
                                 collate_fn=collator_cliff_pkl,
                                 drop_last=drop_last_flag(len(test_dataset), bsz))

        return train_loader, test_loader

    def _get_net(self):
        model = Scage(mode=config['mode'], atom_names=CompoundKit.atom_vocab_dict.keys(),
                      atom_embed_dim=config['model']['atom_embed_dim'],
                      num_kernel=config['model']['num_kernel'], layer_num=config['model']['layer_num'],
                      num_heads=config['model']['num_heads'],
                      atom_FG_class=nfg() + 1,
                      hidden_size=config['model']['hidden_size'], num_tasks=1).cuda()
        model_name = 'model.pth'
        if self.config['pretrain_model_path'] != 'None':
            model_path = os.path.join(self.config['pretrain_model_path'], model_name)
            state_dict = torch.load(model_path, map_location='cuda')
            model.model.load_model_state_dict(state_dict['model'])
            print("Loading pretrain model from", os.path.join(self.config['pretrain_model_path'], model_name))
        if GlobalVar.parallel_train:
            model = nn.DataParallel(model)
        return model

    def _get_loss_fn(self):
        loss_type = self.config['loss_type']
        if loss_type == 'bce':
            return bce_loss()
        elif loss_type == 'wb_bce':
            ratio = self.imbalance_ratio
            return bce_loss(weights=[1.0, ratio])
        elif loss_type == 'mse':
            return nn.MSELoss()
        elif loss_type == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError('not supported loss function!')

    def _get_optim(self):
        optim_type = self.config['optim']['type']
        lr = self.config['optim']['init_lr']
        weight_decay = self.config['optim']['weight_decay']
        freezing_layers_regex = []
        freezing_layers = []
        if GlobalVar.freeze_layers > 0 and GlobalVar.use_ckpt:
            for i in range(GlobalVar.freeze_layers):
                freezing_layers_regex.append(f'EncoderAtomList.{i}')
                freezing_layers_regex.append(f'EncoderBondList.{i}')
        for name, param in self.net.named_parameters():
            if GlobalVar.freeze_layers > 0:
                for part_ in freezing_layers_regex:
                    if part_ in name:
                        param.requires_grad = False
                        freezing_layers.append(name)
                        break

        base_params = list(
            map(lambda x: x[1], list(filter(lambda kv: kv[0] not in freezing_layers, self.net.named_parameters()))))
        base_params_names = list(
            map(lambda x: x[0], list(filter(lambda kv: kv[0] not in freezing_layers, self.net.named_parameters()))))

        model_params = [{'params': base_params}]
        for p_name in freezing_layers:
            assert p_name in [p[0] for p in self.net.named_parameters()]
            assert p_name not in base_params_names
        if optim_type == 'adam':
            return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'rms':
            return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
        elif optim_type == 'sgd':
            momentum = self.config['optim']['momentum'] if 'momentum' in self.config['optim'] else 0
            return torch.optim.SGD(model_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError('not supported optimizer!')

    def _get_lr_scheduler(self):
        scheduler_type = self.config['lr_scheduler']['type']
        init_lr = self.config['lr_scheduler']['start_lr']
        warm_up_epoch = self.config['lr_scheduler']['warm_up_epoch']

        if scheduler_type == 'linear':
            return LinearSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                              warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'square':
            return SquareSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                              warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'cos':
            return CosSche(self.config['epochs'], warm_up_end_lr=self.config['optim']['init_lr'], init_lr=init_lr,
                           warm_up_epoch=warm_up_epoch)
        elif scheduler_type == 'None':
            return None
        else:
            raise ValueError('not supported learning rate scheduler!')

    def _step(self, model, batch: Dict[str, Tensor]):
        function_group_type_nmber = GlobalVar.fg_number
        pred_dict: Dict = model(batch)
        pred_y = pred_dict['graph_feature']
        pred_cliff = pred_dict['cliff_feature']
        finger = pred_dict['finger_feature']
        atom_fg = pred_dict['atom_fg']
        function_group_index = batch['function_group_index']

        loss_atom_fg = get_balanced_atom_fg_loss(atom_fg, function_group_index,
                                                         loss_f_atom_fg=F.binary_cross_entropy_with_logits)

        loss_finger = F.binary_cross_entropy_with_logits(finger, batch['morgan2048_fp'].float())

        reg_loss = F.mse_loss(pred_y.view(-1), batch['label_y'].float())
        if GlobalVar.use_cliff_pred_loss:
            cls_loss = F.binary_cross_entropy_with_logits(pred_cliff.view(-1), batch['label_cliff'].float())
            loss = reg_loss + cls_loss * 0.1
        else:
            loss = reg_loss

        return self.calc_mt_loss([loss, loss_finger, loss_atom_fg]), pred_y, pred_cliff

    def _train_step(self):
        self.net.train()
        num_data = 0
        train_loss = 0
        y_pred = []  # list of length batch_size. Items are [float]
        y_true = []  # list of length batch_size. Items are [1] or [-1] for bbbp
        cliff_list = []
        for _, batch in tqdm(enumerate(self.train_loader)):
            self.optim.zero_grad()
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None

            loss, pred_y, pred_cliff = self._step(self.net, batch)
            train_loss += loss.item()
            self.writer.add_scalar('train_loss', loss, global_step=self.optim_steps)

            y_pred.extend(pred_y.cpu().detach().numpy())
            y_true.extend(batch['label_y'].cpu().numpy())
            cliff_list.extend(batch['label_cliff'].cpu().numpy())

            loss.backward()
            self.optim.step()
            num_data += 1
            self.optim_steps += 1

        train_loss /= num_data
        torch.cuda.empty_cache()
        rmse = calc_rmse(y_true, y_pred)
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_pred, y_test=y_true, cliff_mols_test=cliff_list)

        return train_loss, rmse, rmse_cliff

    def _test_step(self):
        self.net.eval()
        y_pred = []
        y_true = []
        cliff_list = []
        test_loss = 0
        num_data = 0
        for batch in self.test_loader:
            batch = {key: value.to('cuda') for key, value in batch.items()
                     if value is not None and not isinstance(value, list)}
            batch['edge_weight'] = None
            with torch.no_grad():
                loss, pred_y, pred_cliff = self._step(self.net, batch)
            test_loss += loss.item()
            num_data += 1
            y_pred.extend(pred_y.cpu().detach().numpy())
            y_true.extend(batch['label_y'].cpu().numpy())
            cliff_list.extend(batch['label_cliff'].cpu().numpy())

        test_loss /= num_data
        rmse = calc_rmse(y_true, y_pred)
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_pred, y_test=y_true, cliff_mols_test=cliff_list)

        return test_loss, rmse, rmse_cliff

    def save_ckpt(self, epoch):
        checkpoint = {
            "net": self.net.state_dict(),
            'optimizer': self.optim.state_dict(),
            "epoch": epoch,
            'best_metric': self.best_metric,
            'optim_steps': self.optim_steps
        }
        path = os.path.join(self.writer.log_dir, 'checkpoint')
        os.makedirs(path, exist_ok=True)
        torch.save(checkpoint, os.path.join(self.writer.log_dir, 'checkpoint', 'model_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        if GlobalVar.use_ckpt:
            print(f'load model from {load_pth}')
            checkpoint = torch.load(load_pth, map_location='cuda')['model']
            new_model_dict = self.net.state_dict()
            new_model_keys = set(list(new_model_dict.keys()))

            pretrained_dict = {'.'.join(k.split('.')): v for k, v in checkpoint.items()}
            pretrained_keys = set(list('.'.join(k.split('.')) for k in checkpoint.keys()))
            is_dp = model_is_dp(pretrained_keys)
            if is_dp:
                # rmv 'module.' prefix
                pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
                pretrained_keys = set(list(k[7:] for k in pretrained_keys))
                # only update the same keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_keys}
            diff_, same_ = new_model_keys - pretrained_keys, new_model_keys & pretrained_keys
            not_used_keys = pretrained_keys - new_model_keys
            new_model_dict.update(pretrained_dict)
            self.net.load_state_dict(new_model_dict)

        self.start_epoch = 1
        self.optim_steps = 0
        self.best_metric = np.inf
        self.writer = SummaryWriter('../train_result/{}/{}_{}_{}_{}_{}'.format(
            'finetune_result_cliff', config['task_name'], config['seed'],
            config['optim']['init_lr'],
            config['batch_size'], datetime.now().strftime('%b%d_%H:%M:%S')
        ))

    def train(self):
        write_record(self.txtfile, self.config)
        self.net = self.net.to('cuda')

        test_rmse_list = []
        test_rmse_cliff_list = []
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.optim, i)
            # print("Epoch {} cur_lr {}".format(i, self.optim.param_groups[1]['lr']))
            train_loss, train_rmse, train_rmse_cliff = self._train_step()
            test_loss, test_rmse, test_rmse_cliff = self._test_step()
            test_rmse_list.append(test_rmse)
            test_rmse_cliff_list.append(test_rmse_cliff)
            # print(GlobalVar.dist_bar)

            if self.best_metric > test_rmse:
                self.best_metric = test_rmse

                model_dict = {'model': self.net.state_dict()}
                if RoutineControl.save_best_ckpt:
                    torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))

            if RoutineControl.save_best_ckpt and i % 10 == 0:
                torch.save(model_dict, os.path.join(self.writer.log_dir, f'model_{i}.pth'))

            print(f'train_loss:{train_loss} train_rmse:{train_rmse} train_rmse_cliff:{train_rmse_cliff}\t'
                  f'test_loss:{test_loss} test_rmse:{test_rmse} test_rmse_cliff:{test_rmse_cliff}')

            write_record(self.txtfile,
                         f'train_loss:{train_loss} train_rmse:{train_rmse} train_rmse_cliff:{train_rmse_cliff}\t'
                         f'test_loss:{test_loss} test_rmse:{test_rmse} test_rmse_cliff:{test_rmse_cliff}')

        best_test_rmse = np.min(test_rmse_list)
        best_test_rmse_cliff = np.min(test_rmse_cliff_list)
        print(f'best_test_rmse:{best_test_rmse} best_test_rmse_cliff:{best_test_rmse_cliff}')
        write_record(self.txtfile, f'best_test_rmse:{best_test_rmse} best_test_rmse_cliff:{best_test_rmse_cliff}')

def parse_args():
    parser = argparse.ArgumentParser(description='parameters of finetuning SCAGE on Activity cliffs datasets')

    parser.add_argument('--task', type=str, default='CHEMBL231_Ki', help='task name (default: CHEMBL231_Ki)')
    parser.add_argument('--dataroot', type=str, default="./data/cliff/cliff_pkl/", help='data root')
    parser.add_argument('--cliff_closs', action='store_true', default=False,
                        help='add active cliff classification tasks')

    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--seed', default=8, type=int, help='seed (default: 8)')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size (default: 64)')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate (default: 0.0005)')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='weight decay of the optimizer (default: 0.0001)')
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='dropout rate (default: 0.1)')
    parser.add_argument('--dataloader_num_workers', default=4, type=int,
                        help='number of processes loading the dataset (default: 4)')
    parser.add_argument('--emdedding_dim', default=512, type=int,
                        help='embedding dimensions for atomic features (default: 512)')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='hidden layer dimension in transformer (default: 256)')
    parser.add_argument('--layer_num', default=6, type=int,
                        help='number of transformer layers (default: 6)')
    parser.add_argument('--num_heads', default=16, type=int,
                        help='for controlling long attention spans (default: 16)')
    parser.add_argument('--optim_type', default='adam', type=str, choices=['adam', 'rms', 'sgd'],
                        help='optimizer type (default: adam)')
    parser.add_argument('--dist_bar', nargs='+', type=int, default=[20, 50],
                        help="selecting distance bars")
    parser.add_argument('--gpus', type=str, default='0', help='gpu ids')
    parser.add_argument('--ckpt', type=str, default='pretrain', help='checkpoint path')

    args = parser.parse_args()
    return args



def main(config):
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.ckpt != 'None':
        args.checkpoint = f'./weights/pretrain/{args.ckpt}.pth'
        GlobalVar.use_ckpt = True
        if args.ckpt == 'None':
            GlobalVar.use_ckpt = False
            print("GlobalVar.use_ckpt set to False, because args.ckpt == 'None'")

    user = 'cliff'
    config['userconfig'][user]['dataset_dir'] = args.dataroot
    config = config_current_user(user, config)
    config = config_dataset_form('pkl', config)
    config['task_name'] = args.task
    config['batch_size'] = args.batch_size
    config['seed'] = args.seed
    config['dropout'] = args.dropout
    DEFAULTS.DROP_RATE = args.dropout
    config['dataloader_num_workers'] = args.dataloader_num_workers
    config['optim']['init_lr'] = args.lr
    config['checkpoint'] = args.checkpoint
    config['optim']['type'] = args.optim_type.split('/')[0] if '/' in args.optim_type else args.optim_type
    config['optim']['weight_decay'] = float(args.optim_type.split('=')[1]) if '/' in args.optim_type else args.weight_decay
    config['optim']['momentum'] = float(args.optim_type.split('=')[1]) if '/' in args.optim_type else 0
    set_seed(config['seed'])
    GlobalVar.use_cliff_pred_loss = args.cliff_closs
    print('GlobalVar.use_cliff_pred_loss=', GlobalVar.use_cliff_pred_loss)
    RoutineControl.save_best_ckpt = False
    if len(args.dist_bar) == 2:
        set_dist_bar_two(args.dist_bar[0], args.dist_bar[1])
    elif len(args.dist_bar) == 1:
        GlobalVar.dist_bar = [args.dist_bar[0]]
    elif len(args.dist_bar) == 0:
        set_dist_bar_three(args.dist_bar[0], args.dist_bar[1], args.dist_bar[2])
    print(GlobalVar.dist_bar)
    if GlobalVar.dist_bar is None:
        sys.exit(0)
    GlobalVar.embedding_style = "more"
    GlobalVar.transformer_dim = args.emdedding_dim
    GlobalVar.ffn_dim = args.hidden_dim
    GlobalVar.num_heads = args.num_heads
    GlobalVar.freeze_layers = 0
    if GlobalVar.freeze_layers < 0:
        GlobalVar.use_ckpt = False
        print("GlobalVar.use_ckpt set to False, because GlobalVar.freeze_layers < 0")
    GlobalVar.parallel_train = False

    config['model']['atom_embed_dim'] = GlobalVar.transformer_dim
    config['model']['bond_embed_dim'] = GlobalVar.transformer_dim
    config['model']['hidden_size'] = GlobalVar.ffn_dim
    config['model']['num_heads'] = GlobalVar.num_heads
    config['model']['layer_num'] = args.layer_num
    config['fg_num_'] = nfg() + 1
    config['freeze_layers'] = GlobalVar.freeze_layers
    config['loss_style'] = GlobalVar.loss_style
    print_debug(dumps_json(config))

    trainer = Trainer(config, path)
    trainer.train()


if __name__ == '__main__':
    path = Path(pdir) / "config" / "config_finetune_cliff.yaml"
    config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    main(config)
