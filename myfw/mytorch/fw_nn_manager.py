import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim

from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
# ----
import chainer
# from chainer.training import extensions
# from chainer import reporter
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

import chainer_pytorch_migration as cpm
import chainer_pytorch_migration.ignite

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
# ----
from myfw.base_nn_manager import base_nn_manager
from myfw.mytorch.create_supervised_chainer import create_supervised_chainer_trainer, create_supervised_chainer_evaluator

class fw_nn_manager(base_nn_manager):
    def __init__(self):
        super(fw_nn_manager, self).__init__()
        self.out = 'result_pytorch'
        self.out_model = 'models_pytorch'

        self.loss_fn = None
        self.lazy = True

    def set_param(self):

        if self.model_framework_type == 'pytorch': 
            # torch.cuda.is_available()
            # self.device = 'cuda:0'
            if self.device >= 0:
                self.device = 'cuda'
            else:
                self.device = 'cpu'
            self.device = torch.device(self.device)
        elif self.model_framework_type == 'chainer': 
            self.device = chainer.get_device(self.device)
        else:
            raise ValueError

        torch.manual_seed(self.seed)

        super(fw_nn_manager, self).set_param()

    # DataLoader, Iterator
    def set_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)


    # model
    def set_model(self):
        if self.model_framework_type == 'pytorch': 
            if self.lazy:
                dummy_input = self.train_loader.dataset[0][0]
                self.model(dummy_input)
            # self.model.to(self.device)
        elif self.model_framework_type == 'chainer': 
            self.model.to_device(self.device)
            self.device.use()
            self.model(self.device.xp.zeros((1,) + self.input_size).astype('f'))
            self.model = cpm.LinkAsTorchModel(self.model)
        else:
            raise ValueError

        if self.model_framework_type == 'pytorch': 
            self.loss_fn = F.nll_loss
        elif self.model_framework_type == 'chainer': 
            self.loss_fn = softmax_cross_entropy
        else:
            raise ValueError

    # Optimizer
    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        # self.optimizer = optim.Adam(self.model.parameters())
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)
        self.optimizer.step()

    # Ignite, Updater
    def set_trainer(self):
        trainer_fn = None
        
        if self.model_framework_type == 'pytorch':
            trainer_fn = create_supervised_trainer
        elif self.model_framework_type == 'chainer':
            trainer_fn = create_supervised_chainer_trainer
        else:
            raise ValueError
        
        self.trainer = trainer_fn(self.model, self.optimizer, self.loss_fn, device=self.device)

    # event handler
    '''
    def set_chainer_event_handler(self):

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.eval_interval))
        def report_loss(engine):
            # reporter.report({'loss':engine.state.output})
            reporter.report({'train/loss':engine.state.output})

        self.optimizer.target = self.model
        self.trainer.out = self.out

        if self.model_framework_type == 'chainer':
            # (Not Supported)Evaluator
            pass

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ExponentialShift('lr', 0.9))
        # (Not Implemented)InverseShift
        # (Not Implemented)LinearShift
        # (Not Implemented)MultistepShift
        # (Not Implemented)PolynomialShift
        # (Not Implemented)StepShift
        # (Not Implemented)WarmupShift

        # cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.MicroAverage('loss', 'lr', 'mav'))

        # (Not Implemented)cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.VariableStatisticsPlot(self.model))
        # (Not Implemented)cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ParameterStatistics(self.model, trigger=(self.eval_interval, 'epoch')))

        # observe_value 
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.observe_lr())

        # 'iteration', 'mav', 'model/fc2/weight/grad/percentile/1'
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.PrintReport(['epoch', 'elapsed_time', 'train/loss']), call_before_training=self.call_before_training)
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.LogReport(trigger=(self.log_interval, 'epoch')), call_before_training=self.call_before_training)
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.PlotReport(['train/loss'], 'epoch', filename='loss.png'), call_before_training=self.call_before_training)
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ProgressBar())

        # writer=writer
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.snapshot(n_retains=self.retain_num), trigger=(self.log_interval, 'epoch'))

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.FailOnNonNumber())

        if self.model_framework_type == 'chainer':
            # (Not Supported)DumpGraph
            # (Not Supported)unchain_variables
            pass
    '''

    # event handler
    def set_ppe_manager(self):

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.eval_interval))
        def report_loss(engine):
            ppe.reporting.report({'train/loss':engine.state.output})

        # manager.extend
        my_extensions = [
            # (Not Implemented)ExponentialShift
            # (Not Implemented)InverseShift
            # (Not Implemented)LinearShift
            # (Not Implemented)MultistepShift
            # (Not Implemented)PolynomialShift
            # (Not Implemented)StepShift
            # (Not Implemented)WarmupShift

            # extensions.MicroAverage('loss', 'lr', 'mav'),

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

            # (Not Implemented)FailOnNonNumber
        ]

        if self.model_framework_type == 'pytorch':
            my_extensions += [extensions.IgniteEvaluator(self.evaluator, self.valid_loader, self.model, progress_bar=True)] # エラー回避
        elif self.model_framework_type == 'chainer':
            # (Not Supported)DumpGraph
            # (Not Supported)unchain_variables
            pass
        else:
            raise ValueError

        models = {'main': self.model}
        optimizers = {'main': self.optimizer}
        self.ppe_manager = ppe.training.IgniteExtensionsManager(self.trainer, models, optimizers, self.max_epochs, extensions=my_extensions, out_dir=self.out)

    # event_handler
    def set_event_handler(self):

        # (Not Implemented)call_before_training

        if self.model_framework_type == 'pytorch': 
            # self.evaluator = create_supervised_evaluator(self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(F.nll_loss)}, device=self.device)
            self.evaluator = create_supervised_evaluator(self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss_fn)}, device=self.device)
        elif self.model_framework_type == 'chainer': 
            # self.evaluator = create_supervised_chainer_evaluator(self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(self.loss_fn)}, device=self.device) # エラー回避
            pass
        else:
            raise ValueError

        '''
        train_history = {'loss':[], 'accuracy':[]}
        valid_history = {'loss':[], 'accuracy':[]}

        @self.trainer.on(Events.EPOCH_COMPLETED(every=self.eval_interval))
        def eval_train(engine):
            evaluator.run(self.train_loader)
            metrics = evaluator.state.metrics
            loss = metrics['loss']
            accuracy = metrics['accuracy']
            train_history['loss'].append(loss)
            train_history['accuracy'].append(accuracy)
            print("train : epoch : {}, loss : {:.2f}, accuracy : {:.2f}".format(engine.state.epoch, loss, accuracy))

        @self.trainer.on(Events.EPOCH_COMPLETED(every=self.eval_interval))
        def eval_valid(engine):
            evaluator.run(self.valid_loader)
            metrics = evaluator.state.metrics
            loss = metrics['loss']
            accuracy = metrics['accuracy']
            valid_history['loss'].append(loss)
            valid_history['accuracy'].append(accuracy)
            print("valid : epoch : {}, loss : {:.2f}, accuracy : {:.2f}".format(engine.state.epoch, loss, accuracy))

        checkpointer = ModelCheckpoint(
            # './models',
            self.out_model,
            'MNIST',
            save_interval=None,
            n_saved=self.retain_num, 
            create_dir=True, 
            save_as_state_dict=True,
            require_empty=False,
        )
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED(every=self.log_interval), checkpointer, {'MNIST': self.model})
        '''

        # self.set_chainer_event_handler()
        self.set_ppe_manager()

    def resume(self):
        if self.resume_filepath is not None:
            # cpm.ignite.load_chainer_snapshot(self.trainer, self.optimizer, self.resume_filepath)

            state = torch.load(self.resume_filepath)
            self.ppe_manager.load_state_dict(state)

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=self.max_epochs)
