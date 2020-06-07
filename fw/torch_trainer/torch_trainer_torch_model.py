import os

import torch
import torch.nn.functional as F
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy
# ----
from fw.torch_trainer.torch_trainer import torch_trainer
from fw.torch_model.MLP import MLP

class torch_trainer_torch_model(torch_trainer):

    def __init__(self):
        super(torch_trainer_torch_model, self).__init__()

        self.lazy = True

        self.out = os.path.join(self.out, 'pytorch_model')

        self.collate_fn = None
        self.create_trainer_fn = create_supervised_trainer
        self.create_evaluator_fn = create_supervised_evaluator
        self.loss_fn = F.nll_loss
        self.accuracy_class = Accuracy

    # device
    def set_device(self):
        # torch.cuda.is_available()
        # self.device = 'cuda:0'
        if self.device >= 0:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)

    # model
    def set_model(self):
        self.model = MLP()

        if self.lazy:
            dummy_input = self.train_loader.dataset[0][0]
            self.model(dummy_input)
        # self.model.to(self.device)

    # post Optimizer
    def post_set_optimizer(self):
        self.optimizer.step()
