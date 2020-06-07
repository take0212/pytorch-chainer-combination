import os

import chainer
import chainer.links as L
from chainer.dataset import convert
from chainer import training
from chainer.training import extensions
# ----
from fw.chainer_trainer.chainer_trainer import chainer_trainer
from fw.chainer_model.MLP import MLP

class chainer_trainer_chainer_model(chainer_trainer):

    def __init__(self):
        super(chainer_trainer_chainer_model, self).__init__()

        self.out = os.path.join(self.out, 'chainer_model')

        self.updater_class = training.updaters.StandardUpdater
        self.converter = convert.concat_examples


    # device
    def set_device(self):
        self.device = chainer.get_device(self.device)


    # model
    def set_model(self):
        model = MLP()
        self.model = L.Classifier(model)
        self.classifier = self.model # same
        self.classifier.to_device(self.device)
        self.device.use()


    def set_target(self):
        self.target = self.classifier


    def set_additonal_event_handler(self):
        self.trainer.extend(extensions.DumpGraph('main/loss'))
        # (Not Implemented) unchain_variables
