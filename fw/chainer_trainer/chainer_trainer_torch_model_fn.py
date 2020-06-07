from chainer.dataset import convert
from chainer.training.updaters import StandardUpdater
# ----
import torch


def torch_accuracy(y, t, ignore_label=None):
    correct = (t.eq(y.argmax(dim=1).long())).sum()
    # return correct / len(t)
    return correct.double() / len(t)


def tensor_converter(batch, device):
    in_arrays = convert.concat_examples(batch, device)
    in_arrays = (
        torch.tensor(in_arrays[0]).to(device),
        torch.tensor(in_arrays[1]).long().to(device),
    )

    return in_arrays


class TorchStandardUpdater(StandardUpdater):

    def update_core(self):
        iterator = self._iterators["main"]
        batch = iterator.next()
        in_arrays = convert._call_converter(
            self.converter, batch, self.input_device)

        optimizer = self._optimizers["main"]
        loss_func = self.loss_func or optimizer.target

        # The graph should be traversed in PyTorch
        # the optimizer holds a chainerized model
        # that can't be executed
        loss = loss_func(*in_arrays)

        # We need to do the backward step ourselves instead
        # of relying in optimizer because it does calls that
        # the torch API does not support
        loss.backward()

        optimizer.update()
