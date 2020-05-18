from chainer.dataset import convert
from chainer.training.updaters import StandardUpdater
# ----
import torch
# import torch.nn.functional as torchF

def tensor_converter(batch, device):

    in_arrays = convert.concat_examples(batch, device)
    # in_arrays_tensor = (torch.tensor(in_arrays[0]).to('cuda:0'), torch.tensor(in_arrays[1]).long().to('cuda:0'))
    in_arrays_tensor = (torch.tensor(in_arrays[0]).to(device), torch.tensor(in_arrays[1]).long().to(device))

    return in_arrays_tensor

class TorchStandardUpdater(StandardUpdater):

    def update_core(self):
        iterator = self._iterators['main']
        batch = iterator.next()
        in_arrays = convert._call_converter(self.converter, batch, self.input_device)

        optimizer = self._optimizers['main']
        # loss_func = self.loss_func or optimizer.target
        loss_func = self.loss_func
        model = optimizer.target.module

        # x = torch.tensor(in_arrays[0]).to('cuda:0')
        # y = model(x)
        y = model(in_arrays[0])

        # t = torch.tensor(in_arrays[1]).long().to('cuda:0')
        t = in_arrays[1]

        # loss = torchF.nll_loss(y, t)
        loss = loss_func(y, t)
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
