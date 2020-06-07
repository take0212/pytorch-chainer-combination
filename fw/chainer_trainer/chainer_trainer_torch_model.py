import os

import cupy
import chainer.links as L
import chainer_pytorch_migration as cpm
# ----
import torch.nn.functional as torchF
# ----
from fw.chainer_trainer.chainer_trainer import chainer_trainer
from fw.torch_model.MLPWrapper import MLPWrapper
from fw.chainer_trainer.chainer_trainer_torch_model_fn import TorchStandardUpdater, tensor_converter, torch_accuracy

class chainer_trainer_torch_model(chainer_trainer):

    def __init__(self):
        super(chainer_trainer_torch_model, self).__init__()

        self.out = os.path.join(self.out, 'pytorch_model')

        self.updater_class = TorchStandardUpdater
        self.converter = tensor_converter


    # device
    def set_device(self):
        self.device = cupy.cuda.Device(self.device)


    # model
    def set_model(self):
        # torch_model = MLPWrapper(lazy=False)
        torch_model = MLPWrapper()
        torch_model = torch_model.cuda()

        dummy_input = self.train_dataset[0]
        dummy_input = self.converter([dummy_input], self.device.id)
        torch_model(dummy_input[0])

        self.model = cpm.TorchModule(torch_model)
        self.model.to_gpu(self.device)

        # We create a classifier over the PyTorch model,
        # since it is the one that will be called
        self.classifier = L.Classifier(torch_model, lossfun=torchF.nll_loss, accfun=torch_accuracy)


    # pre trainer
    def pre_set_trainer(self):
        # Hack for the trainer to register the correct model in the reporter
        # model -> classifier
        self.optimizer.target = self.classifier


    # post trainer
    def post_set_trainer(self):
        # For the attributes to be correctly updated
        # classifier -> model
        self.optimizer.target = self.model


    def set_target(self):
        self.target = {"main": self.classifier}
