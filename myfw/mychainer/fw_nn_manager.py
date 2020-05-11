# import
import chainer
from chainer import iterators
from chainer import optimizers
from chainer import training
from chainer.training import extensions
# ----
from myfw.base_nn_manager import base_nn_manager

class fw_nn_manager(base_nn_manager):
    def __init__(self):
        super(fw_nn_manager, self).__init__()
        self.out = 'result_chainer'
        self.out_model = 'models_chainer'

    def set_param(self):
        if self.model_framework_type == 'pytorch': 

            # import torch
            # if self.device >= 0:
            #     self.pytorch_device = 'cuda'
            # else:
            #     self.pytorch_device = 'cpu'
            # self.pytorch_device = torch.device(self.pytorch_device)
            self.pytorch_device = self.device

        elif self.model_framework_type == 'chainer': 
            pass
        else:
            raise ValueError

        self.device = chainer.get_device(self.device)

        if self.retain_num is None:
            self.retain_num = -1

        # (Not Implemented)seed

        super(fw_nn_manager, self).set_param()

    # DataLoader, Iterator
    def set_dataloader(self):
        # if self.model_framework_type == 'chainer': 
        self.train_loader = iterators.SerialIterator(self.train_dataset, self.batch_size)
        self.valid_loader = iterators.SerialIterator(self.valid_dataset, self.batch_size, repeat=False, shuffle=False)
        # elif self.model_framework_type == 'pytorch': 
        #     from torch.utils.data import DataLoader
        #     self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        #     self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)
        # else:
        #     raise ValueError

    # model
    def set_model(self):

        if self.model_framework_type == 'chainer': 

            self.model.to_device(self.device)
            self.device.use()

        elif self.model_framework_type == 'pytorch': 

            import chainer_pytorch_migration as cpm
            self.model.cuda()
            self.model = cpm.TorchModule(self.model)
            self.model.to_gpu(self.pytorch_device)
            # self.model.to_gpu(self.device)

        else:
            raise ValueError

    # Optimizer
    def set_optimizer(self):
        # self.optimizer = optimizers.SGD(lr=self.lr)
        self.optimizer = optimizers.MomentumSGD(lr=self.lr, momentum = self.momentum)
        # self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)

    # Ignite, Updater
    def set_trainer(self):

        if self.model_framework_type == 'chainer': 
            updater = training.updaters.StandardUpdater(self.train_loader, self.optimizer, device=self.device)
        elif self.model_framework_type == 'pytorch':
            from myfw.mychainer.PyTorchStandardUpdater import PyTorchStandardUpdater
            updater = PyTorchStandardUpdater(self.train_loader, self.optimizer, device=self.device)
        else:
            raise ValueError

        self.trainer = training.Trainer(updater, (self.max_epochs, 'epoch'), out=self.out)

    # event handler
    def set_event_handler(self):

        # (Not Implemented)Evaluator(train)
        self.trainer.extend(extensions.Evaluator(self.valid_loader, self.model, device=self.device), trigger=(self.eval_interval, 'epoch'), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.ProgressBar())

        self.trainer.extend(extensions.observe_lr())

        # self.trainer.extend(extensions.MicroAverage('loss', 'lr', 'mav'))

        self.trainer.extend(extensions.LogReport(trigger=(self.log_interval, 'epoch')), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.FailOnNonNumber())

        self.trainer.extend(extensions.ExponentialShift('lr', rate=0.9))
        # (Not Implemented)InverseShift
        # (Not Implemented)LinearShift
        # (Not Implemented)MultistepShift
        # (Not Implemented)PolynomialShift
        # (Not Implemented)StepShift
        # (Not Implemented)WarmupShift

        self.trainer.extend(extensions.ParameterStatistics(self.model, trigger=(self.eval_interval, 'epoch')))

        self.trainer.extend(extensions.VariableStatisticsPlot(self.model))

        self.trainer.extend(extensions.PrintReport(['epoch', 'elapsed_time', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'), call_before_training=self.call_before_training)
        self.trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'), call_before_training=self.call_before_training)

        self.trainer.extend(extensions.snapshot(n_retains=self.retain_num), trigger=(self.log_interval, 'epoch'))

        self.trainer.extend(extensions.DumpGraph('main/loss'))

        # (Note Implemented) unchain_variables

    def resume(self):
        if self.resume_filepath is not None:
            self.chainer.serializers.load_npz(self.resume_filepath, self.trainer)

    def run(self):
        self.trainer.run()
