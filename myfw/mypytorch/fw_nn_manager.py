# import
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
from myfw.base_nn_manager import base_nn_manager

class fw_nn_manager(base_nn_manager):
    def __init__(self):
        super(fw_nn_manager, self).__init__()
        self.out = 'result_pytorch'
        self.out_model = 'models_pytorch'

    def set_param(self):

        if self.model_framework_type == 'chainer': 
            import chainer
            self.chainer_device = chainer.get_device(self.device)
            pass
        elif self.model_framework_type == 'pytorch': 
            pass
        else:
            raise ValueError

        # torch.cuda.is_available()
        # self.device = 'cuda:0'
        if self.device >= 0:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.device = torch.device(self.device)

        torch.manual_seed(self.seed)

        super(fw_nn_manager, self).set_param()

    # DataLoader, Iterator
    def set_dataloader(self):
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False)


    # model
    def set_model(self):
        if self.model_framework_type == 'pytorch': 
            pass
        elif self.model_framework_type == 'chainer': 

            import chainer_pytorch_migration as cpm
            self.model.to_device(self.chainer_device)
            self.model(self.chainer_device.xp.zeros((1,) + self.input_size).astype('f'))
            self.model = cpm.LinkAsTorchModel(self.model)

        else:
            raise ValueError

    # Optimizer
    def set_optimizer(self):
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        # self.optimizer = optim.Adam(self.model.parameters())
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.lr)

    # Ignite, Updater
    def set_trainer(self):
        self.trainer = create_supervised_trainer(self.model, self.optimizer, F.nll_loss, device=self.device)

    # event handler
    def set_chainer_event_handler(self):

        from chainer import reporter
        from chainer.training import extensions
        import chainer_pytorch_migration as cpm
        import chainer_pytorch_migration.ignite

        @self.trainer.on(Events.ITERATION_COMPLETED(every=self.eval_interval))
        def report_loss(engine):
            reporter.report({'loss':engine.state.output})

        self.optimizer.target = self.model
        self.trainer.out = self.out

        # (Note Supported)Evaluator

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ProgressBar())

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.observe_lr())

        # cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.MicroAverage('loss', 'lr', 'mav'))

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.LogReport(trigger=(self.log_interval, 'epoch')), call_before_training=self.call_before_training)

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.FailOnNonNumber())

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ExponentialShift('lr', 0.9))
        # (Not Implemented)InverseShift
        # (Not Implemented)LinearShift
        # (Not Implemented)MultistepShift
        # (Not Implemented)PolynomialShift
        # (Not Implemented)StepShift
        # (Not Implemented)WarmupShift

        # (Not Implemented)cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.ParameterStatistics(self.model, trigger=(self.eval_interval, 'epoch')))

        # (Not Implemented)cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.VariableStatisticsPlot(self.model))

        # 'iteration', 'mav', 'model/fc2/weight/grad/percentile/1'
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.PrintReport(['epoch', 'elapsed_time', 'loss']), call_before_training=self.call_before_training)

        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.PlotReport(['loss'], 'epoch', filename='loss.png'), call_before_training=self.call_before_training)

        # writer=writer
        cpm.ignite.add_trainer_extension(self.trainer, self.optimizer, extensions.snapshot(n_retains=self.retain_num), trigger=(self.log_interval, 'epoch'))

        # (Not Supported)DumpGraph
        # (Not Supported)unchain_variables

    # event_handler
    def set_event_handler(self):

        # (Not Implemented)call_before_training

        evaluator = create_supervised_evaluator(self.model, metrics={'accuracy': Accuracy(), 'loss': Loss(F.nll_loss)}, device=self.device)

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

        self.set_chainer_event_handler()

    def resume(self):
        if self.resume_filepath is not None:
            import chainer_pytorch_migration as cpm
            cpm.ignite.load_chainer_snapshot(self.trainer, self.optimizer, self.resume_filepath)

    def run(self):
        self.trainer.run(self.train_loader, max_epochs=self.max_epochs)
