import os

import chainer
from chainer.datasets import mnist
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from fw.base_trainer import base_trainer

class chainer_trainer(base_trainer):

    def __init__(self):
        super(chainer_trainer, self).__init__()

        self.out = os.path.join(self.out, 'chainer_trainer')

        self.updater_class = None
        self.converter = None

        self.classifier = None
        self.target = None


    def set_param(self):
        super(chainer_trainer, self).set_param()

        if self.retain_num is None:
            self.retain_num = -1

        # (Not Implemented)seed


    # device
    def set_device(self):
        raise NotImplementedError


    # Dataset
    def set_dataset(self):
        if self.dataset == 'mnist':
            self.train_dataset, self.valid_dataset = mnist.get_mnist()
        else:
            raise ValueError


    # DataLoader, Iterator
    def set_dataloader(self):
        self.train_loader = iterators.SerialIterator(self.train_dataset, self.batch_size)
        self.valid_loader = iterators.SerialIterator(self.valid_dataset, self.batch_size, repeat=False, shuffle=False)


    # model
    def set_model(self):
        raise NotImplementedError


    # Optimizer
    def set_optimizer(self):
        # self.optimizer = optimizers.SGD(lr=self.lr)
        self.optimizer = optimizers.MomentumSGD(lr=self.lr, momentum = self.momentum)
        # self.optimizer = optimizers.Adam()

        self.optimizer.setup(self.model)


    # pre trainer
    def pre_set_trainer(self):
        pass


    # post trainer
    def post_set_trainer(self):
        pass


    # Ignite, Updater
    def set_trainer(self):
        updater = self.updater_class(self.train_loader, self.optimizer, converter=self.converter, device=self.device, loss_func=self.classifier)
        self.pre_set_trainer()
        self.trainer = training.Trainer(updater, (self.max_epochs, 'epoch'), out=self.out)
        self.post_set_trainer()


    def set_target(self):
        raise NotImplementedError


    def set_additonal_event_handler(self):
        pass


    # event handler
    def set_event_handler(self):

        self.set_target()

        # (Not Implemented)Evaluator(train)
        self.trainer.extend(extensions.Evaluator(self.valid_loader, self.target, converter=self.converter, device=self.device,), trigger=(self.eval_interval, 'epoch'), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.ProgressBar())

        self.trainer.extend(extensions.observe_lr())

        # self.trainer.extend(extensions.MicroAverage('loss', 'lr', 'mav'))

        self.trainer.extend(extensions.LogReport(trigger=(self.log_interval, 'epoch')), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.FailOnNonNumber())

        # self.trainer.extend(extensions.ExponentialShift('lr', rate=0.9))
        self.trainer.extend(extensions.ExponentialShift('lr', rate=0.99, init=self.lr*10.0))
        # (Not Implemented)InverseShift
        # (Not Implemented)LinearShift
        # (Not Implemented)MultistepShift
        # (Not Implemented)PolynomialShift
        # (Not Implemented)StepShift
        # (Not Implemented)WarmupShift

        self.trainer.extend(extensions.ParameterStatistics(self.model, trigger=(self.eval_interval, 'epoch')))

        self.trainer.extend(extensions.VariableStatisticsPlot(self.model))

        self.trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'validation/main/loss', 'validation/main/accuracy', 'elapsed_time']), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'), call_before_training=self.call_before_training)
        self.trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.snapshot(n_retains=self.retain_num), trigger=(self.log_interval, 'epoch'))

        self.set_additonal_event_handler()

    def resume(self):
        if self.resume_filepath is not None:
            chainer.serializers.load_npz(self.resume_filepath, self.trainer)

    def run(self):
        self.trainer.run()
