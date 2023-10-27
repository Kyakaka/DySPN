from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from argparse import ArgumentParser
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split
from DySPN.base import Model
import settings_kitti_s as settings
from dataset.kittidc import KITTIDC
import dc_metric
from loss.sloss import SLoss, L1Loss, L2Loss
import warnings
import cv2
from PIL import Image
import os
import shutil
from utils import summary
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
# torch.set_float32_matmul_precision('medium')
def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(".git*", ".idea*", "*pycache*", "*index_files*", "*lightning_logs*", "*dataset*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


class Lit_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.basenet = Model(iteration=settings.iteration,
                             num_neighbor=settings.num_neighbor,
                             mode=settings.spn_module,
                             norm_depth=settings.norm_depth,
                             bm=settings.basemodel,
                             res=settings.resnet,
                             stodepth=settings.sto_depth,
                             norm_layer='bn',
                             shuffle_up=False)

        self.metric = dc_metric.DC_Metric(settings.eval_range)

        self.loss = SLoss(depth_range=settings.norm_depth)

    def on_train_start(self) -> None:
        if self.local_rank == 0 and self.current_epoch == 0:
            path_backup = '{}/{}'.format(self.logger.log_dir, "code")
            os.makedirs(path_backup, exist_ok=True)
            backup_source_code(path_backup)

    def on_train_epoch_start(self) -> None:
        if self.local_rank == 0:
            tensorboard = self.logger.experiment
            for name, param in self.basenet.named_parameters():
                tensorboard.add_histogram(tag=name + "_data", values=param.data, global_step=self.current_epoch)

    def training_step(self, sample, batch_idx):
        # Forward through the network
        image, sparse_depth, ground_truth = \
            sample['rgb'], sample['dep'], sample['gt']
        output = self.basenet(rgb0=image, dep=sparse_depth)
        loss = self.loss(output['pred'], ground_truth)
        self.log("loss",loss,prog_bar=True,sync_dist=True)
        return loss

    def validation_step(self, sample, batch_idx):
        # Forward through the network
        image, sparse_depth, ground_truth = \
            sample['rgb'], sample['dep'], sample['gt']
        output = self.basenet(rgb0=image, dep=sparse_depth)
        if settings.val_output and self.local_rank==0 and batch_idx==0:
            path_output = '{}/epoch{:04d}'.format(self.logger.log_dir, self.current_epoch)
            os.makedirs(path_output, exist_ok=True)
            summary(sample,output,path_output,settings)
        # metric
        self.metric(ground_truth, output['pred'])
        return 0

    def on_validation_epoch_end(self, ):
        rmse, mae, irmse, imae, rel, del1, del2, del3 = self.metric.compute()
        self.log('RMSE', rmse, sync_dist=True)
        self.log('MAE', mae, sync_dist=True)
        self.log('iRMSE', irmse, sync_dist=True)
        self.log('iMAE', imae, sync_dist=True)
        self.log('rel', rel, sync_dist=True)
        self.log('del1', del1, sync_dist=True)
        self.log('del2', del2, sync_dist=True)
        self.log('del3', del3, sync_dist=True)
        self.metric.reset()

    def test_step(self, sample, batch_idx):
        # Forward through the network
        image, sparse_depth, ground_truth = \
            sample['rgb'], sample['dep'], sample['gt']
        output = self.basenet(rgb0=image, dep=sparse_depth)
        if self.local_rank == 0:
            path_output = '{}/output'.format(self.logger.log_dir)
            os.makedirs(path_output, exist_ok=True)
            path_save_pred = '{}/{:010d}.png'.format(path_output, batch_idx)
            pred = output['pred'][0, 0, :, :].cpu().detach().numpy()
            pred = (pred*256.0).astype(np.uint16)
            # cv2.imwrite(path_save_pred,pred)
            pred = Image.fromarray(pred)
            pred.save(path_save_pred)

    def configure_optimizers(self):
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers, i.e. BatchNorm2d()
        for v in self.basenet.modules():
            for p_name, p in v.named_parameters(recurse=False):
                if p_name == 'bias':  # bias (no decay)
                    g[2].append(p)
                elif p_name == 'weight' and isinstance(v, bn):  # weight (no decay)
                    g[1].append(p)
                else:
                    g[0].append(p)  # weight (with decay)

        optimizer = torch.optim.Adam(g[2], lr=settings.learning_rates)  # adjust beta1 to momentum
        optimizer.add_param_group(
            {'params': g[0], 'weight_decay': settings.w_weight_decay})  # add g0 with weight_decay
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})  # add g1 (BatchNorm2d weights)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, settings.step, 0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def cli_main():
    pl.seed_everything(settings.seed)

    # ------------
    # data
    # ------------

    train_dataloader = torch.utils.data.DataLoader(
        KITTIDC('train'),
        batch_size=settings.n_batch,
        shuffle=True,
        num_workers=settings.n_thread,
        drop_last=False,
        persistent_workers=True)
    val_dataloader = torch.utils.data.DataLoader(
        KITTIDC('val'),
        batch_size=1,
        shuffle=False,
        num_workers=settings.n_thread,
        drop_last=False,
        persistent_workers=True)
    test_dataloader = torch.utils.data.DataLoader(
        KITTIDC('test'),
        batch_size=1,
        shuffle=False,
        num_workers=settings.n_thread,
        drop_last=False,
        persistent_workers=True)
    # ------------
    # model
    # ------------

    if settings.pretrain_weight is not None:
        model = Lit_Model.load_from_checkpoint(settings.pretrain_weight)
    else:
        model = Lit_Model()
    lr_monitor_cb = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    save_model_cb = pl.callbacks.ModelCheckpoint(monitor='RMSE',
                                                 mode='min',
                                                 filename="{epoch}-{RMSE:.4f}",
                                                 save_last=True)
    trainer = pl.Trainer(accelerator='gpu',
                         devices=settings.n_device,
                         max_epochs=settings.epochs,
                         precision=16,
                         callbacks=[lr_monitor_cb, save_model_cb],
                         sync_batchnorm=True,
                         strategy='ddp_find_unused_parameters_true',
                         # detect_anomaly=True,
                         )

    if settings.test_only == False:
        # ------------
        # training
        # ------------
        trainer.fit(model, train_dataloader, val_dataloader,ckpt_path=settings.resume_weight)
        # ------------
        # validating
        # ------------
        trainer.validate(model,val_dataloader)
    else:
        # ------------
        # testing
        # ------------
        trainer.test(model,test_dataloader)



if __name__ == '__main__':
    cli_main()
