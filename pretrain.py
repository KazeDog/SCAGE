import argparse
import itertools

import loguru

from _config import *
from _config import get_audit_atom_fg_frequency_arr
from naive_fg import all_possible_fg_nums

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import yaml
import numpy as np
from datetime import datetime
from tqdm import tqdm
from echo_logger import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

from data_process.function_group_constant import nfg
from utils.public_util import set_seed
from utils.loss_util import NTXentLoss, get_balanced_atom_fg_loss, get_focal_loss
from utils.scheduler_util import *
from utils.fg_inspect import get_spatial_pos_frequency_arr, get_pair_distances_frequency_arr, get_angle_frequency_arr

from datasets.dataloader import PretrainDataset
from models.scage import Scage
from data_process.compound_tools import CompoundKit
from data_process.data_collator import collator_pretrain_pkl_bin

from utils.global_var_util import GlobalVar, RoutineControl, \
    set_dist_bar_two, set_dist_bar_three
from utils.userconfig_util import config_current_user, config_dataset_form
from utils.metric_util import compute_atom_fg_metric_cuda_tensor, compute_sp_metric, \
    compute_pair_distances_metric, compute_angles_metric, compute_finger_metric

import warnings
from typing import Dict, List

warnings.filterwarnings("ignore")


class PreTrainer(object):
    def __init__(self, config, file_path):
        self.config = config

        self.train_loader, self.test_loader = self.get_data_loaders()
        self.model = self._get_net()
        self.optim = self._get_optim()
        self.lr_scheduler = self._get_lr_scheduler()

        if config['checkpoint']:
            print('loading check point')
            self.load_ckpt(self.config['checkpoint'])
        else:
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_loss = np.inf
            self.best_auc = -np.inf
            self.writer = SummaryWriter('../train_result/{}/{}/{}_{}_{}'.format(
                'pretrain_result', config['pretrain_name'], config['task_name'], config['seed'],
                datetime.now().strftime('%b%d_%H_%M_%S')))
        self.txtfile = os.path.join(self.writer.log_dir, 'record.txt')
        self.batch_considered = 200

        self.loss_init = torch.zeros(GlobalVar.loss_num, self.batch_considered, device='cuda')
        self.loss_last = torch.zeros(GlobalVar.loss_num, self.batch_considered // 10, device='cuda')
        self.loss_last2 = torch.zeros(GlobalVar.loss_num, self.batch_considered // 10, device='cuda')
        self.cur_loss_step = torch.zeros(1, dtype=torch.long, device='cuda')
        copyfile(file_path, self.writer.log_dir)

    def get_data_loaders(self):
        path_ = self.config['root']
        dataset = PretrainDataset(self.config['root'])
        print('all_dataset_num:', len(dataset))
        # train_dataset = Subset(dataset, range(0, len(dataset) - 2000))
        # test_dataset = Subset(dataset, range(len(dataset) - 2000, len(dataset)))
        train_dataset = Subset(dataset, range(0, len(dataset) - 100))
        test_dataset = Subset(dataset, range(len(dataset) - 100, len(dataset)))
        # train_dataset = Subset(dataset, range(0, 9900))
        # test_dataset = Subset(dataset, range(9900, 10000))
        print('train_dataset_num:', len(train_dataset))
        print('test_dataset_num:', len(test_dataset))
        num_workers = self.config['dataloader_num_workers']
        bsz = self.config['batch_size']
        train_loader = DataLoader(train_dataset,
                                  batch_size=bsz,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  collate_fn=collator_pretrain_pkl_bin,
                                  pin_memory=True,
                                  drop_last=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=bsz,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 collate_fn=collator_pretrain_pkl_bin,
                                 pin_memory=True,
                                 drop_last=True)

        return train_loader, test_loader

    def _get_net(self):
        model = Scage(mode=config['mode'], atom_names=CompoundKit.atom_vocab_dict.keys(),
                      atom_embed_dim=config['model']['atom_embed_dim'],
                      num_kernel=config['model']['num_kernel'], layer_num=config['model']['layer_num'],
                      num_heads=config['model']['num_heads'],
                      atom_FG_class=nfg() + 1,
                      hidden_size=config['model']['hidden_size'], num_tasks=None).cuda()

        model_name = 'model.pth'
        if self.config['pretrain_model_path'] != 'None':
            model_path = os.path.join(self.config['pretrain_model_path'], model_name)
            state_dict = torch.load(model_path, map_location='cuda')
            model.model.load_model_state_dict(state_dict['model'])
            print("Loading pretrain model from", os.path.join(self.config['pretrain_model_path'], model_name))
        if GlobalVar.parallel_train:
            model = nn.DataParallel(model)
        return model

    def calc_mt_loss(self, loss_list):
        loss_list = torch.stack(loss_list).cuda()
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

    def _get_loss_fn(self):
        loss_type = self.config['loss']['type']
        if loss_type == 'NTXentLoss':
            return NTXentLoss(self.config['batch_size'], **self.config['loss']['param'])

    def _get_optim(self):
        optim_type = self.config['optim']['type']
        lr = self.config['optim']['init_lr']
        weight_decay = self.config['optim']['weight_decay']

        if optim_type == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            return optimizer
        elif optim_type == 'rms':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            return optimizer
        elif optim_type == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
            return optimizer
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

    def _step(self, model, batch):
        function_group_type_nmber = GlobalVar.fg_number
        pred_dict: Dict = model(batch)

        loss_list = []
        metric_result = {}
        if 'finger' in GlobalVar.pretrain_task:
            atom_finger_feature = pred_dict['atom_finger_feature']

            loss_atom_finger = F.binary_cross_entropy_with_logits(atom_finger_feature,
                                                                  batch['morgan2048_fp'].float())
            metric_result['atom_finger_feature'] = atom_finger_feature

            loss_list += [loss_atom_finger]

        if 'fg' in GlobalVar.pretrain_task:
            atom_fg = pred_dict['atom_fg']
            function_group_index = batch['function_group_index']
            function_group_bond_index = batch['function_group_bond_index']

            if GlobalVar.balanced_atom_fg_loss:
                loss_atom_fg = get_balanced_atom_fg_loss(atom_fg, function_group_index,
                                                         loss_f_atom_fg=F.binary_cross_entropy_with_logits)
            else:
                loss_atom_fg = F.binary_cross_entropy_with_logits(atom_fg.view(-1, function_group_type_nmber),
                                                                  (function_group_index + 0.0).view(-1,
                                                                                                    function_group_type_nmber))

            mask_atom_fg = (function_group_index != -1)
            atom_fg = atom_fg[mask_atom_fg]
            function_group_index = function_group_index[mask_atom_fg]

            # loss_list += [loss_atom_fg, loss_bond_fg]
            loss_list += [loss_atom_fg]
            metric_result['atom_fg'] = atom_fg
            # metric_result['bond_fg'] = bond_fg
            metric_result['function_group_index'] = function_group_index
            metric_result['function_group_bond_index'] = function_group_bond_index

        if 'sp' in GlobalVar.pretrain_task:
            spatial_pos_pred = pred_dict['spatial_pos_pred']
            spatial_pos = batch['spatial_pos']

            mask_spatial_pos = (spatial_pos != -1)
            spatial_pos_pred = spatial_pos_pred[mask_spatial_pos]
            spatial_pos = spatial_pos[mask_spatial_pos]
            loss_spatial_pos = get_focal_loss(spatial_pos_pred.reshape(-1, 21), spatial_pos.reshape(-1).long(),
                                              get_spatial_pos_frequency_arr(), 'cuda')
            # loss_spatial_pos = F.cross_entropy(spatial_pos_pred.reshape(-1, 11), spatial_pos.reshape(-1).long())

            loss_list += [loss_spatial_pos]
            metric_result['spatial_pos_pred'] = spatial_pos_pred
            metric_result['spatial_pos'] = spatial_pos
        if 'angle' in GlobalVar.pretrain_task:
            angle_pred = pred_dict['angle_pred']
            bond_angles = batch['bond_angles_bin']

            mask_bond_angles = (bond_angles != -5)
            bond_angles = bond_angles[mask_bond_angles]
            loss_bond_angles = get_focal_loss(angle_pred.reshape(-1, 20), bond_angles.reshape(-1).long(),
                                              get_angle_frequency_arr(), 'cuda')
            # loss_bond_angles = F.cross_entropy(angle_pred.reshape(-1, 12), bond_angles.reshape(-1).long())

            loss_list += [loss_bond_angles]
            metric_result['angle_pred'] = angle_pred
            metric_result['bond_angles'] = bond_angles

        if GlobalVar.use_calc_mt_loss:
            GlobalVar.loss_num = len(loss_list)
            loss = self.calc_mt_loss(loss_list)
        else:
            loss = sum(loss_list)

        return loss, pred_dict, metric_result

    def _train_step(self):
        self.model.train()
        num_data = 0
        loss_all = 0
        batch_index = 0
        total_step = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, leave=False, ascii=True, total=total_step)
        for batch_idx, batch in enumerate(progress_bar):
            self.optim.zero_grad()

            batch = {key: value.to('cuda') for key, value in batch.items()
                         if value is not None and not isinstance(value, list)}
            loss, pred_dict, metric_result = self._step(self.model, batch)
            __DEBUG__FLAG__ = RoutineControl.debug_train_loss_and_auc
            tqdm_meg = f'train_loss:{loss:.5f}, ' if not __DEBUG__FLAG__ else f'train_loss:{loss:.3f}, '
            if __DEBUG__FLAG__:
                with torch.no_grad():
                    if 'fg' in GlobalVar.pretrain_task:
                        fg_acc, atom_fg_auc = compute_atom_fg_metric_cuda_tensor(
                            metric_result['atom_fg'].view(-1, GlobalVar.fg_number),
                            metric_result['function_group_index'].view(-1, GlobalVar.fg_number))
                        tqdm_meg += (f'fg_acc:{fg_acc:.2f},'
                                     f'atom_fg_auc:{atom_fg_auc:.2f},')
                    if 'sp' in GlobalVar.pretrain_task:
                        sp_acc, sp_auc = compute_sp_metric(metric_result['spatial_pos_pred'].reshape(-1, 21),
                                                           metric_result['spatial_pos'].reshape(-1))
                        tqdm_meg += f'sp_acc:{sp_acc:.2f},sp_auc:{sp_auc:.2f},'
                    if 'pair_distance' in GlobalVar.pretrain_task:
                        pair_distences_acc, pair_distences_auc = compute_pair_distances_metric(
                            metric_result['pair_distences_pred'].reshape(-1, 30),
                            metric_result['pair_distances'].reshape(-1).long())
                        tqdm_meg += f'dist_acc:{pair_distences_acc:.2f},dist_auc:{pair_distences_auc:.2f},'
                    if 'angle' in GlobalVar.pretrain_task:
                        angle_acc, angle_auc = compute_angles_metric(metric_result['angle_pred'].reshape(-1, 20),
                                                                     metric_result['bond_angles'].reshape(-1).long())
                        tqdm_meg += f'angle_acc:{angle_acc:.2f},angle_auc:{angle_auc:.2f}'
            loss.backward()

            loss_all += loss.item()

            progress_bar.set_description(tqdm_meg, refresh=True)
            self.optim.step()
            num_data += 1
            self.optim_steps += 1

            batch_index += 1

        loss_mean = loss_all / len(self.train_loader)
        return loss_mean

    def _valid_step(self, valid_loader):
        self.model.eval()
        if 'finger' in GlobalVar.pretrain_task:
            atom_finger_pred = torch.tensor([], device='cuda')
            atom_finger_true = torch.tensor([], device='cuda')
        if 'fg' in GlobalVar.pretrain_task:
            atom_fg_pred = torch.tensor([], device='cuda')
            atom_fg_true = torch.tensor([], device='cuda')
        if 'sp' in GlobalVar.pretrain_task:
            sp_pred = torch.tensor([], device='cuda')
            sp_true = torch.tensor([], device='cuda')
        if 'angle' in GlobalVar.pretrain_task:
            bond_angle_pred = torch.tensor([], device='cuda')
            bond_angle_true = torch.tensor([], device='cuda')
        valid_loss = 0
        num_data = 0
        progress_bar = tqdm(valid_loader, leave=False, ascii=True)
        for batch in progress_bar:
            batch = {key: value.to('cuda') for key, value in batch.items()
                         if value is not None and not isinstance(value, list)}
            with torch.no_grad():
                loss, pred_dict, metric_result = self._step(self.model, batch)

                if 'finger' in GlobalVar.pretrain_task:
                    atom_finger_pred = torch.cat(
                        (atom_finger_pred, metric_result['atom_finger_feature'].reshape(-1, 2048)),
                        dim=0)
                    atom_finger_true = torch.cat((atom_finger_true, batch['morgan2048_fp'].reshape(-1, 2048)),
                                                 dim=0)

                if 'fg' in GlobalVar.pretrain_task:
                    atom_fg_pred = torch.cat((atom_fg_pred, metric_result['atom_fg'].view(-1, GlobalVar.fg_number)),
                                             dim=0)
                    atom_fg_true = torch.cat(
                        (atom_fg_true, metric_result['function_group_index'].view(-1, GlobalVar.fg_number)), dim=0)

                if 'sp' in GlobalVar.pretrain_task:
                    sp_pred = torch.cat((sp_pred, metric_result['spatial_pos_pred'].reshape(-1, 21)), dim=0)
                    sp_true = torch.cat((sp_true, metric_result['spatial_pos'].reshape(-1)), dim=0)

                if 'angle' in GlobalVar.pretrain_task:
                    bond_angle_pred = torch.cat((bond_angle_pred, metric_result['angle_pred'].reshape(-1, 20)), dim=0)
                    bond_angle_true = torch.cat((bond_angle_true, metric_result['bond_angles'].reshape(-1)), dim=0)

            valid_loss += loss.item()
            num_data += 1

        valid_loss /= num_data
        valid_msg = ''
        valid_auc_all = 0
        if 'finger' in GlobalVar.pretrain_task:
            finger_acc_all, finger_auc_all = compute_finger_metric(atom_finger_pred, atom_finger_true.long())
            valid_auc_all += finger_auc_all
            valid_msg += f'finger_acc_all:{finger_acc_all:.4f}, finger_auc_all:{finger_auc_all:.4f} '
        if 'fg' in GlobalVar.pretrain_task:
            fg_acc_all, atom_fg_auc_all = compute_atom_fg_metric_cuda_tensor(atom_fg_pred, atom_fg_true.long())
            valid_auc_all = valid_auc_all + atom_fg_auc_all
            valid_msg += (f'fg_acc:{fg_acc_all:.4f}, '
                          f'atom_fg_auc_all:{atom_fg_auc_all:.4f}, ')
        if 'sp' in GlobalVar.pretrain_task:
            sp_acc_all, sp_auc_all = compute_sp_metric(sp_pred, sp_true.long())
            valid_auc_all += sp_auc_all
            valid_msg += f'sp_acc_all:{sp_acc_all:.4f}, sp_auc_all:{sp_auc_all:.4f}, '
        if 'angle' in GlobalVar.pretrain_task:
            angle_acc_all, angle_auc_all = compute_angles_metric(bond_angle_pred, bond_angle_true.long())
            valid_auc_all += angle_auc_all
            valid_msg += f'angle_acc_all:{angle_acc_all:.4f}, angle_auc_all:{angle_auc_all:.4f} '

        return valid_loss, valid_msg, valid_auc_all

    def save_ckpt(self, epoch):
        model_dict = {
            'model': self.model.state_dict()
        }
        checkpoint = {
            "net": model_dict,
            'optim': self.optim.state_dict(),
            "epoch": epoch,
            'best_loss': self.best_loss,
            'optim_steps': self.optim_steps
        }
        path = os.path.join(self.writer.log_dir, 'checkpoint')
        os.makedirs(path, exist_ok=True)
        loguru.logger.info(f'Save checkpoint at epoch {epoch}, path: {path}')
        torch.save(checkpoint, os.path.join(path, 'model_{}.pth'.format(epoch)))

    def load_ckpt(self, load_pth):
        checkpoint = torch.load(load_pth, map_location='cuda')
        # print(checkpoint)
        self.writer = SummaryWriter(os.path.dirname(load_pth))
        self.model.load_state_dict(checkpoint['net']['model'])
        self.optim.load_state_dict(checkpoint['optim'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.optim_steps = checkpoint['optim_steps']

    def train(self):
        # print(self.config)
        write_record(self.txtfile, self.config)
        self.model = self.model.to('cuda')
        for i in range(self.start_epoch, self.config['epochs'] + 1):
            if self.config['lr_scheduler']['type'] in ['cos', 'square', 'linear']:
                self.lr_scheduler.adjust_lr(self.optim, i)

            model_loss = self._train_step()
            test_loss, valid_msg, valid_auc_all = self._valid_step(self.test_loader)

            self.writer.add_scalar('test_loss', test_loss, global_step=i)

            if GlobalVar.use_calc_mt_loss:
                if valid_auc_all > self.best_auc:
                    self.best_auc = valid_auc_all
                    print(f'Current best auc:{self.best_auc}. Save checkpoint.')
                    write_record(self.txtfile,
                                 f'Current best auc:{self.best_auc}. Save checkpoint.')
                    if self.config['DP']:
                        model_dict = {
                            'model': self.model.module.state_dict()
                        }
                        torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
                    else:
                        model_dict = {
                            'model': self.model.state_dict()
                        }
                        torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))

            else:
                if model_loss < self.best_loss:
                    self.best_loss = model_loss
                    print(f'Current best loss:{self.best_loss}. Save checkpoint.')
                    write_record(self.txtfile,
                                 f'Current best loss:{self.best_loss}. Save checkpoint.')
                    if self.config['DP']:
                        model_dict = {
                            'model': self.model.module.state_dict()
                        }
                        torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))
                    else:
                        model_dict = {
                            'model': self.model.state_dict()
                        }
                        torch.save(model_dict, os.path.join(self.writer.log_dir, 'model.pth'))

            if i % self.config['save_ckpt'] == 0:
                self.save_ckpt(i)

            print(f'Epoch:{i} model_loss:{model_loss:.4f}, auc_all:{valid_auc_all:.4f}, {valid_msg} ')
            write_record(self.txtfile,
                         f'Epoch:{i} model_loss:{model_loss:.4f}, auc_all:{valid_auc_all:.4f}, {valid_msg}')
            loguru.logger.info(f"Clearing python memory cache...")
            torch.cuda.empty_cache()



# set GlobalVar.parallel_train to True if we have more than one GPU in visible devices.

RoutineControl.debug_train_loss_and_auc = False
get_audit_atom_fg_frequency_arr()
path = "config/config_pretrain.yaml"
config = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
GlobalVar.pretrain_dataset_is_from_pkl_file_directly = False
config = config_current_user('pretrain', config)
config = config_dataset_form('pkl', config)

def parse_args():

    parser = argparse.ArgumentParser(description='parameters of pretraining SCAGE')

    parser.add_argument('--dataroot', type=str, default="./data/pretrain/toy.pkl",
                        help='data root')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--seed', default=8, type=int, help='seed (default: 8)')
    parser.add_argument('--batch_size', default=40, type=int, help='batch size (default: 40)')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate (default: 0.00005)')
    parser.add_argument('--weight_decay', default=0.0001, type=float,
                        help='weight decay of the optimizer (default: 0.0001)')
    parser.add_argument('--dataloader_num_workers', default=24, type=int,
                        help='number of processes loading the dataset (default: 24)')
    parser.add_argument('--emdedding_dim', default=512, type=int,
                        help='embedding dimensions for atomic features (default: 512)')
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help='hidden layer dimension in transformer (default: 256)')
    parser.add_argument('--layer_num', default=6, type=int,
                        help='number of transformer layers (default: 6)')
    parser.add_argument('--num_heads', default=16, type=int,
                        help='for controlling long attention spans (default: 16)')
    parser.add_argument('--dist_bar', nargs='+', type=int, default=[20, 50],
                        help="selecting distance bars")
    parser.add_argument('--pretrain_task', nargs='+', type=str, default=['finger', 'sp', 'angle', 'fg'],
                        help="selecting pretraining tasks")
    parser.add_argument('--gpus', type=str, default='0', help='gpu ids')

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    GlobalVar.parallel_train = True if "CUDA_VISIBLE_DEVICES" in os.environ and \
            len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1 else False
    config['pretrain_name'] = 'pretrain'
    config['task_name'] = '_'

    config['epochs'] = args.epochs
    config['seed'] = args.seed
    config['batch_size'] = args.batch_size
    config['optim']['init_lr'] = args.lr
    config['optim']['weight_decay'] = args.weight_decay
    config['dataloader_num_workers'] = args.dataloader_num_workers
    config['model']['atom_embed_dim'] = args.emdedding_dim
    config['model']['bond_embed_dim'] = args.emdedding_dim
    config['model']['hidden_size'] = args.hidden_dim
    config['model']['layer_num'] = args.layer_num
    config['model']['num_heads'] = args.num_heads
    config['pretrain_task'] = args.pretrain_task
    config['mode'] = 'pretrain_bin'

    set_seed(config['seed'])
    GlobalVar.embedding_style = "more"
    GlobalVar.transformer_dim = args.emdedding_dim
    GlobalVar.ffn_dim = args.hidden_dim
    GlobalVar.layer_num = args.layer_num
    GlobalVar.num_heads = args.num_heads
    GlobalVar.use_calc_mt_loss = True
    if len(args.pretrain_task) == 1:
        GlobalVar.use_calc_mt_loss = False
        loguru.logger.warning("You are using only one task, so we will not use mt loss.")
    GlobalVar.balanced_atom_fg_loss = True

    GlobalVar.pretrain_task = args.pretrain_task
    GlobalVar.loss_num = GlobalVar.get_loss_num()
    if len(args.dist_bar) == 2:
        set_dist_bar_two(args.dist_bar[0], args.dist_bar[1])
    elif len(args.dist_bar) == 1:
        GlobalVar.dist_bar = [args.dist_bar[0]]
    elif len(args.dist_bar) == 0:
        set_dist_bar_three(args.dist_bar[0], args.dist_bar[1], args.dist_bar[2])
    print_debug(dumps_json(config))
    loguru.logger.info(f"parallel_train={GlobalVar.parallel_train}, using GPU id: {os.environ['CUDA_VISIBLE_DEVICES']}")
    loguru.logger.info("Real batch size on each GPU: " + str(
        args.batch_size // len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
    loguru.logger.info(f"GlobalVar.dist_bar, {GlobalVar.dist_bar}")
    trainer = PreTrainer(config, path)
    trainer.train()



if __name__ == '__main__':
    main()