from chainer.training.updaters import StandardUpdater
from chainer.dataset import convert

class PyTorchStandardUpdater(StandardUpdater):

    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = convert._call_converter(
            self.converter, batch, self.input_device)

        optimizer = self._optimizers['main']

        # loss_func = self.loss_func or optimizer.target

        model = optimizer.target.module
        y = model(in_arrays[0])
        t = in_arrays[1]

        import torch.nn.functional as F
        loss = F.nll_loss(y, t)
        loss.backward()

        optimizer.update()

        # if isinstance(in_arrays, tuple):
        #     optimizer.update(loss_func, *in_arrays)
        # elif isinstance(in_arrays, dict):
        #     optimizer.update(loss_func, **in_arrays)
        # else:
        #     optimizer.update(loss_func, in_arrays)

        if self.auto_new_epoch and iterator.is_new_epoch:
            optimizer.new_epoch(auto=True)

