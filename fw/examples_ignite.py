import argparse

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
import chainerx
import numpy
from chainer.dataset import concat_examples
import torch
import ignite
import chainer_pytorch_migration as cpm
import chainer_pytorch_migration.ignite
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from chainer.functions.loss import softmax_cross_entropy

# 6/4/2020 added
from chainer.functions.evaluation import accuracy as chainer_accuracy
from ignite.metrics.accuracy import Accuracy
from ignite.metrics.metric import reinit__is_reduced


import matplotlib
matplotlib.use('Agg')


# 6/4/2020 added
def chainer_converter(args):
    n_args = [arg for arg in args]
    for i,arg in enumerate(args):
        if isinstance(arg, cpm.parameter._ChainerTensor):
            n_args[i] = arg._variable
    return n_args


# 6/4/2020 added
class ChainerAccuracy(Accuracy):
    @reinit__is_reduced
    def update(self, output):
        y_pred, y = chainer_converter(output)

        correct = chainer_accuracy.accuracy(y_pred, y )

        self._num_correct += correct.item() * y.shape[0]
        self._num_examples += y.shape[0]


# Network definition
class MLP(chainer.Chain):

    def __init__(self, n_in, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            #self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            #self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            #self.l3 = L.Linear(None, n_out)  # n_units -> n_out
            self.l1 = L.Linear(n_in, n_units)  # n_in -> n_units
            self.l2 = L.Linear(n_units, n_units)  # n_units -> n_units
            self.l3 = L.Linear(n_units, n_out)  # n_units -> n_out

    def forward(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=5, # 20
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', '-f', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', type=str,
                        help='Resume the training from snapshot')
    parser.add_argument('--autoload', action='store_true',
                        help='Automatically load trainer snapshots in case'
                        ' of preemption or other temporary system failure')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    # model = MLP(784, args.unit, 10)
    model = MLP(784, args.unit, 10)
    model.to_device(device)
    device.use()

    torched_model = cpm.LinkAsTorchModel(model)

    # Setup an optimizer
    #optimizer = chainer.optimizers.Adam()
    #optimizer.setup(model)
    optimizer = torch.optim.Adam(torched_model.parameters())
    # 6/3/2020 added
    optimizer = cpm.parameter.Optimizer(optimizer)
    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist()

    #train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    #test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
    #                                             repeat=False, shuffle=False)
    # 6/1/2020 changed
    # collate_fn = cpm.ignite.collate_to_numpy
    collate_fn = cpm.ignite.collate_to_array
    train_loader = torch.utils.data.DataLoader(train, shuffle=True, batch_size=args.batchsize, pin_memory=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test, shuffle=False, batch_size=args.batchsize, pin_memory=True, collate_fn=collate_fn)

    # Set up a trainer
    #updater = training.updaters.StandardUpdater(
    #    train_iter, optimizer, device=device)
    #trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    torch_device = torch.device('cpu')

    def loss_fn(*args):
        # 6/4/2020 added
        args = chainer_converter(args)
        return softmax_cross_entropy.softmax_cross_entropy(*args)
    
    def prepare_batch(batch, device=None, non_blocking=False):
        return batch

    trainer = ignite.engine.create_supervised_trainer(
        torched_model, optimizer, loss_fn, device=torch_device, prepare_batch=prepare_batch)

    # Evaluate the model with the test dataset for each epoch
    #trainer.extend(extensions.Evaluator(test_iter, model, device=device),
    #               call_before_training=True)
    evaluator = ignite.engine.create_supervised_evaluator(
        torched_model,
        metrics={
            # 2020/6/4 changed
            # 'accuracy': ignite.metrics.Accuracy(),
            'accuracy': ChainerAccuracy(),
            'loss': ignite.metrics.Loss(loss_fn),
        },
        prepare_batch=prepare_batch,
        device=torch_device)

    @trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
    def validation(engine):
        evaluator.run(test_loader)
        # 2020/6/4 chnaged
        average_accuracy = evaluator.state.metrics['accuracy']
        average_loss = evaluator.state.metrics['loss']
        print(average_accuracy, average_loss)
        # print(average_loss)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    # TODO(niboshi): Temporarily disabled for chainerx. Fix it.
    #if device.xp is not chainerx:
    #    trainer.extend(extensions.DumpGraph('main/loss'))

    # Take a snapshot for each specified epoch
    #frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    # Take a snapshot each ``frequency`` epoch, delete old stale
    # snapshots and automatically load from snapshot files if any
    # files are already resident at result directory.
    #trainer.extend(extensions.snapshot(n_retains=1, autoload=args.autoload),
    #               trigger=(frequency, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    #trainer.extend(extensions.LogReport(), call_before_training=True)

    # Save two plot images to the result dir
    #trainer.extend(
    #    extensions.PlotReport(['main/loss', 'validation/main/loss'],
    #                          'epoch', file_name='loss.png'),
    #    call_before_training=True)
    #trainer.extend(
    #    extensions.PlotReport(
    #        ['main/accuracy', 'validation/main/accuracy'],
    #        'epoch', file_name='accuracy.png'),
    #    call_before_training=True)

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    #trainer.extend(extensions.PrintReport(
    #    ['epoch', 'main/loss', 'validation/main/loss',
    #     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']),
    #    call_before_training=True)

    # Print a progress bar to stdout
    #trainer.extend(extensions.ProgressBar())

    #if args.resume is not None:
    #    # Resume from a snapshot (Note: this loaded model is to be
    #    # overwritten by --autoload option, autoloading snapshots, if
    #    # any snapshots exist in output directory)
    #    chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    #trainer.run()
    trainer.run(train_loader, max_epochs=args.epoch)


if __name__ == '__main__':
    main()
