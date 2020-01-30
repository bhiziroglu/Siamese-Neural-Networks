import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data import MNIST

import pytorch_lightning as pl
from pytorch_lightning import LightningModule


class Model(LightningModule):
    """
    Sample model to show how to define a template
    """

    def __init__(self, hparams):
        """
        Pass in parsed HyperOptArgumentParser to the model
        :param hparams:
        """
        # init superclass
        super(Model, self).__init__()
        self.hparams = hparams

        self.batch_size = hparams.batch_size

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        """
        Layout model
        :return:
        """
        self.c_d1 = nn.Linear(in_features=self.hparams.in_features,
                              out_features=self.hparams.out_features)

        self.c_d2 = nn.Linear(in_features=self.hparams.out_features,
                              out_features=self.hparams.out_features)

        self.out = nn.Linear(in_features=self.hparams.out_features,
                              out_features=1)

    # ---------------------
    # TRAINING
    # ---------------------
    def _forward(self, im):
        im = im.squeeze(1)
        im = im.view(-1, 28*28)
        x = self.c_d1(im)
        x = F.relu(x)
        x = self.c_d2(x)
        return x

    def forward(self, im1, im2):
        """
        No special modification required for lightning, define as you normally would
        :param x:
        :return:
        """
        out1 = self._forward(im1)
        out2 = self._forward(im2)
        dis = torch.abs(out1 - out2)
        logits = self.out(dis)
        return logits

    def loss(self, logits, labels):
        loss_fn = F.binary_cross_entropy_with_logits(logits, labels)
        return loss_fn

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        im1, im2, same, _ = batch

        pred = self.forward(im1, im2)

        # calculate loss
        loss_val = self.loss(pred.squeeze(1), same)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_val = loss_val.unsqueeze(0)

        tqdm_dict = {'train_loss': loss_val}
        output = OrderedDict({
            'loss': loss_val,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        im1, _, same, target = batch

        preds = []

        for digit in range(10): # 10-way one-shot classification
            current_digit_tensor = self.support_set[digit].repeat(self.batch_size, 1, 1, 1) 
            pred = self.forward(im1, current_digit_tensor).squeeze(1)
            preds.append(pred)
        
        preds = torch.stack(preds)
        predicted_digits = preds.max(0).indices
        loss = self.loss(preds.max(0).values, 1.0*(target == predicted_digits))

        correct = sum(predicted_digits==target)

        val_acc = correct * 1.0 / self.batch_size

        if self.on_gpu:
            val_acc = val_acc.cuda(0)

        # in DP mode (default) make sure if result is scalar, there's another dim in the beginning
        if self.trainer.use_dp or self.trainer.use_ddp2:
            val_acc = val_acc.unsqueeze(0)

        output = OrderedDict({
            'val_acc': val_acc,
            'val_loss': loss,
        })

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def validation_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        val_acc_mean = 0.0
        for output in outputs:

            # reduce manually when using dp
            val_acc = output['val_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                val_acc = torch.mean(val_acc)

            val_acc_mean += val_acc

        val_acc_mean /= len(outputs)
        tqdm_dict = {'val_acc': val_acc_mean}
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': output['val_loss']}
        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        return whatever optimizers we want here
        :return: list of optimizers
        """
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def __dataloader(self, train):
        # init data generators
        transform = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
        # dataset = CustomDataset(transform=transform)
        dataset = MNIST(root='dataset', train=train, transform=transform, download=True)

        if not train:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.support_set = torch.from_numpy(dataset.support_set).float().to(device)

        # when using multi-node (ddp) we need to add the  datasampler
        train_sampler = None
        batch_size = self.hparams.batch_size

        if self.use_ddp:
            train_sampler = DistributedSampler(dataset)

        should_shuffle = train_sampler is None
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=should_shuffle,
            sampler=train_sampler,
            num_workers=0,
            drop_last=True
        )

        return loader

    @pl.data_loader
    def train_dataloader(self):
        logging.info('training data loader called')
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        logging.info('val data loader called')
        return self.__dataloader(train=False)

    @pl.data_loader
    def test_dataloader(self):
        logging.info('test data loader called')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self.hparams
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--in_features', default=28 * 28, type=int)
        parser.add_argument('--out_features', default=4096, type=int)
        parser.add_argument('--hidden_dim', default=2048, type=int)
        parser.add_argument('--drop_prob', default=0.2, type=float)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        # data
        parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)

        # training params (opt)
        parser.add_argument('--optimizer_name', default='adam', type=str)
        parser.add_argument('--batch_size', default=128, type=int)
        return parser