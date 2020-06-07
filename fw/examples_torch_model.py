import chainer
import cupy
import torch
import chainer_pytorch_migration as cpm
from chainer.datasets import mnist
from chainer.training import extensions

"""
Simple example that trains a PyTorch model
using several Chainer components
"""


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc2 = torch.nn.Linear(784, 10)
        self.out = torch.nn.LogSoftmax(dim=1)

    def forward(self, *args, **kwargs):
        x = args[0]
        x = self.fc2(x)
        return self.out(x)

    def namedlinks(self, skipself=False):
        # Hack for the evaluator extension to work
        return []


def tensor_converter(batch, device):
    in_arrays = chainer.dataset.convert.concat_examples(batch, device)
    in_arrays_tensor = (
        torch.tensor(in_arrays[0]).to(device),
        torch.tensor(in_arrays[1]).long().to(device),
    )

    return in_arrays_tensor


class MyUpdater(chainer.training.StandardUpdater):
    def set_model(self, model):
        self.model = model

    def update_core(self):
        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = chainer.dataset.convert._call_converter(
            self.converter, batch, self.input_device
        )
        optimizer = self._optimizers["main"]
        # The graph should be traversed in PyTorch
        # the optimizer holds a chainerized model
        # that can't be executed
        loss = self.model(*in_arrays)
        # We need to do the backward step ourselves instead
        # of relying in optimizer because it does calls that
        # the torch API does not support
        loss.backward()
        optimizer.update()


def accuracy(y, t, ignore_label=None):
    correct = (t.eq(y.argmax(dim=1).long())).sum()
    # return correct / len(t)
    return correct.double() / len(t)

train, test = mnist.get_mnist()

batchsize = 32
device = cupy.cuda.Device(0)

train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, False, False)

model = Net().cuda()
chainer_model = cpm.TorchModule(model)
chainer_model.to_gpu(device)

# We create a classifier over the PyTorch model since
# it is the one that will be called
classifier_model = chainer.links.Classifier(
    model, lossfun=torch.nn.functional.nll_loss, accfun=accuracy
)

# Use a chainer optimizer,
# When using pytorch optimizers, a wrapper should be added
# for it to obbey the chainer optimizer interface
optimizer = chainer.optimizers.MomentumSGD()
optimizer.setup(chainer_model)

# We need a custom updater since we will be constructing
# the computational graph in PyTorch
updater = MyUpdater(
    train_iter,
    optimizer,
    converter=tensor_converter,
    device=device,
    loss_func=torch.nn.functional.nll_loss,
)

# We will call a custom model inside, the optimizer target
# needs to be the chainerized model, but the update_core
# needs access to the torch one wrapped with the classifier
updater.set_model(classifier_model)

# Hack for the trainer to register the correct model in the reporter
optimizer.target = classifier_model
trainer = chainer.training.Trainer(updater, (10, "epoch"), out="mnist_result")
# For the attributes to be correctly updated
optimizer.target = chainer_model


trainer.extend(extensions.LogReport())
trainer.extend(extensions.snapshot(filename="snapshot_epoch-{.updater.epoch}"))
trainer.extend(
    extensions.Evaluator(
        test_iter,
        {"main": classifier_model},
        converter=tensor_converter,
        device=device,
    )
)
trainer.extend(
    extensions.PrintReport(
        [
            "epoch",
            "main/loss",
            "main/accuracy",
            "validation/main/loss",
            "validation/main/accuracy",
            "elapsed_time",
        ]
    )
)
trainer.extend(
    extensions.PlotReport(
        ["main/loss", "validation/main/loss"],
        x_key="epoch",
        file_name="loss.png",
    )
)
trainer.extend(
    extensions.PlotReport(
        ["main/accuracy", "validation/main/accuracy"],
        x_key="epoch",
        file_name="accuracy.png",
    )
)

trainer.run()
