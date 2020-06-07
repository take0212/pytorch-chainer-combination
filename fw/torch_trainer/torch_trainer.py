import os

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import optim
from ignite.metrics import Loss
from ignite.engine import Events
# from ignite.handlers import ModelCheckpoint
# ----
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
# from chainer.training import extensions
# from chainer import reporter
# import chainer_pytorch_migration as cpm
# import chainer_pytorch_migration.ignite
# ----
from fw.base_trainer import base_trainer

class torch_trainer(base_trainer):

    def __init__(self):
        super(torch_trainer, self).__init__()

        self.out = os.path.join(self.out, 'pytorch_trainer')

        self.collate_fn = None
        self.create_trainer_fn = None
        self.create_evaluator_fn = None
        self.loss_fn = None
        self.accuracy_class = None

    def set_param(self):
        super(torch_trainer, self).set_param()

        torch.manual_seed(self.seed)

    #device
    def set_device(self):
        raise NotImplementedError

    # Dataset
    def set_dataset(self):

        if self.dataset == 'mnist':
            root="..\data"
            data_transform = ToTensor()
            self.train_dataset = MNIST(download=True, train=True, root=root, transform=data_transform)
            self.valid_dataset = MNIST(download=False, train=False, root=root, transform=data_transform)
        else:
            raise ValueError

    # DataLoader, Iterator
    def set_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=self.collate_fn)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, collate_fn=self.collate_fn)

    # model
    def set_model(self):
        raise NotImplementedError

    # post Optimizer
    def post_set_optimizer(self):
        raise NotImplementedError

    # Optimizer
    def set_optimizer(self):
        # self.optimizer = optim.Adam(self.model.parameters())
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)

        self.post_set_optimizer()
        
    # Ignite, Updater
    def set_trainer(self):
        self.trainer = self.create_trainer_fn(self.model, self.optimizer, self.loss_fn, device=self.device)

    # set_evaluator
    def set_evaluator(self):
        # (Not Implemented)call_before_training
        self.evaluator = self.create_evaluator_fn(self.model, metrics={'accuracy': self.accuracy_class(), 'loss': Loss(self.loss_fn)}, device=self.device)

    # event handler
    def set_ppe_manager(self):

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.eval_interval))
        def report_loss(engine):
            ppe.reporting.report({'train/loss':engine.state.output})

        # manager.extend
        my_extensions = [
            extensions.VariableStatisticsPlot(self.model),
            extensions.ParameterStatistics(self.model, prefix='model'),

            # observe_value 
            extensions.observe_lr(optimizer=self.optimizer),

            extensions.PrintReport(['epoch', 'elapsed_time', 'lr', 'train/loss', 'val/loss', 'val/accuracy']),
            extensions.LogReport(trigger=(self.log_interval, 'epoch')),
            # 'iteration', 'model/fc2.bias/grad/min'
            extensions.PlotReport(['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
            extensions.PlotReport(['val/accuracy'], 'epoch', filename='accuracy.png'),
            extensions.ProgressBar(),

            extensions.snapshot(n_retains=self.retain_num),

            # (Not Implemented)ExponentialShift
            # (Not Implemented)InverseShift
            # (Not Implemented)LinearShift
            # (Not Implemented)MultistepShift
            # (Not Implemented)PolynomialShift
            # (Not Implemented)StepShift
            # (Not Implemented)WarmupShift

            # extensions.MicroAverage('loss', 'lr', 'mav'),

            # (Not Implemented)FailOnNonNumber

            # (Not Supported)DumpGraph
            # (Not Supported)unchain_variables
        ]

        my_extensions += [extensions.IgniteEvaluator(self.evaluator, self.valid_loader, self.model, progress_bar=True)]

        models = {'main': self.model}
        optimizers = {'main': self.optimizer}
        self.ppe_manager = ppe.training.IgniteExtensionsManager(self.trainer, models, optimizers, self.max_epochs, extensions=my_extensions, out_dir=self.out)

    # event_handler
    def set_event_handler(self):
        self.set_evaluator()
        self.set_ppe_manager()

    def resume(self):
        if self.resume_filepath is not None:
            # cpm.ignite.load_chainer_snapshot(self.trainer, self.optimizer, self.resume_filepath)
            state = torch.load(self.resume_filepath)
            self.ppe_manager.load_state_dict(state)

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=self.max_epochs)

    # event handler
    # def set_chainer_event_handler(self):
    #
    #     @self.trainer.on(Events.ITERATION_COMPLETED(every=self.eval_interval))
    #     def report_loss(engine):
    #         # reporter.report({'loss':engine.state.output})
    #         reporter.report({'train/loss':engine.state.output})
    #
    #     self.optimizer.target = self.model
    #     self.trainer.out = self.out
    #
    #     if self.model_framework_type == 'chainer':
    #         # (Not Supported)Evaluator
    #         pass
    #
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ExponentialShift('lr', 0.9))
    #     # (Not Implemented)InverseShift
    #     # (Not Implemented)LinearShift
    #     # (Not Implemented)MultistepShift
    #     # (Not Implemented)PolynomialShift
    #     # (Not Implemented)StepShift
    #     # (Not Implemented)WarmupShift
    #
    #     # cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.MicroAverage('loss', 'lr', 'mav'))
    #
    #     # (Not Implemented)cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.VariableStatisticsPlot(self.model))
    #     # (Not Implemented)cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ParameterStatistics(self.model, trigger=(self.eval_interval, 'epoch')))
    #
    #     # observe_value 
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.observe_lr())
    #
    #     # 'iteration', 'mav', 'model/fc2/weight/grad/percentile/1'
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.PrintReport(['epoch', 'elapsed_time', 'train/loss']), call_before_training=self.call_before_training)
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.LogReport(trigger=(self.log_interval, 'epoch')), call_before_training=self.call_before_training)
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.PlotReport(['train/loss'], 'epoch', filename='loss.png'), call_before_training=self.call_before_training)
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ProgressBar())
    #
    #     # writer=writer
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.snapshot(n_retains=self.retain_num), trigger=(self.log_interval, 'epoch'))
    #
    #     cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.FailOnNonNumber())
    #
    #     if self.model_framework_type == 'chainer':
    #         # (Not Supported)DumpGraph
    #         # (Not Supported)unchain_variables

    # event_handler
    # def set_event_handler(self):
    #     train_history = {'loss':[], 'accuracy':[]}
    #     valid_history = {'loss':[], 'accuracy':[]}
    # 
    #     @self.trainer.on(Events.EPOCH_COMPLETED(every=self.eval_interval))
    #     def eval_train(engine):
    #         evaluator.run(self.train_loader)
    #         metrics = evaluator.state.metrics
    #         loss = metrics['loss']
    #         accuracy = metrics['accuracy']
    #         train_history['loss'].append(loss)
    #         train_history['accuracy'].append(accuracy)
    #         print("train : epoch : {}, loss : {:.2f}, accuracy : {:.2f}".format(engine.state.epoch, loss, accuracy))
    # 
    #     @self.trainer.on(Events.EPOCH_COMPLETED(every=self.eval_interval))
    #     def eval_valid(engine):
    #         evaluator.run(self.valid_loader)
    #         metrics = evaluator.state.metrics
    #         loss = metrics['loss']
    #         accuracy = metrics['accuracy']
    #         valid_history['loss'].append(loss)
    #         valid_history['accuracy'].append(accuracy)
    #         print("valid : epoch : {}, loss : {:.2f}, accuracy : {:.2f}".format(engine.state.epoch, loss, accuracy))
    # 
    #     checkpointer = ModelCheckpoint(
    #         # './models',
    #         self.out,
    #         'MNIST',
    #         save_interval=None,
    #         n_saved=self.retain_num, 
    #         create_dir=True, 
    #         save_as_state_dict=True,
    #         require_empty=False,
    #     )
    #     self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.log_interval), checkpointer, {'MNIST': self.model})
    # 
    #     self.set_chainer_event_handler()
