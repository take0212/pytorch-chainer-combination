import os

import chainer
import chainer_pytorch_migration as cpm
import chainer_pytorch_migration.ignite
from chainer_pytorch_migration import tensor
# ----
from fw.torch_trainer.torch_trainer import torch_trainer
from fw.torch_trainer.torch_trainer_chainer_model_fn import create_chainer_supervised_trainer, create_chainer_supervised_evaluator, chainer_softmax_cross_entropy, ChainerAccuracy
# from fw.torch_trainer.torch_trainer_chainer_model_fn import chainer_prepare_batch
# ----
from fw.chainer_model.MLP import MLP

class torch_trainer_chainer_model(torch_trainer):

    def __init__(self):
        super(torch_trainer_chainer_model, self).__init__()

        self.out = os.path.join(self.out, 'chainer_model')

        self.collate_fn = cpm.ignite.collate_to_array
        self.create_trainer_fn = create_chainer_supervised_trainer
        self.create_evaluator_fn = create_chainer_supervised_evaluator
        self.loss_fn = chainer_softmax_cross_entropy
        self.accuracy_class = ChainerAccuracy

    # device
    def set_device(self):
        self.device = chainer.get_device(self.device)

    # model
    def set_model(self):
        chainer_model = MLP()
        chainer_model.to_device(self.device)
        self.device.use()

        dummy_input = self.train_dataset[0][0]
        dummy_input = chainer.Variable(tensor.asarray(dummy_input))
        dummy_input.to_device(self.device)
        chainer_model(dummy_input)
        # dummy_input = iter(self.train_loader).next()
        # dummy_input = chainer_prepare_batch(dummy_input, self.device)
        # chainer_model(dummy_input[0][0])

        self.model = cpm.LinkAsTorchModel(chainer_model)

    # post Optimizer
    def post_set_optimizer(self):
        self.optimizer = cpm.parameter.Optimizer(self.optimizer)

