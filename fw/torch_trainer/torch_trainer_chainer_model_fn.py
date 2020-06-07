from ignite.engine.engine import Engine
from ignite.metrics.accuracy import Accuracy
from ignite.metrics.metric import reinit__is_reduced
# ----
import chainer
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy
from chainer.functions.evaluation.accuracy import accuracy
import chainer_pytorch_migration as cpm


def chainer_converter(args):
    n_args = [arg for arg in args]
    for i,arg in enumerate(args):
        if isinstance(arg, cpm.parameter._ChainerTensor):
            n_args[i] = arg._variable
    return n_args


def chainer_softmax_cross_entropy(*args):
    args = chainer_converter(args)
    return softmax_cross_entropy(*args)


class ChainerAccuracy(Accuracy):
    @reinit__is_reduced
    def update(self, output):
        y_pred, y = chainer_converter(output)

        correct = accuracy(y_pred, y )

        self._num_correct += correct.item() * y.shape[0]
        self._num_examples += y.shape[0]


def chainer_prepare_batch(batch, device=None, non_blocking=False):
    x, y = batch

    x_variable = chainer.Variable(x)
    x_variable.to_device(device)

    y_variable = chainer.Variable(y)
    y_variable.to_device(device)

    return (x_variable, y_variable)


def create_chainer_supervised_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              # prepare_batch=_prepare_batch,
                              prepare_batch=chainer_prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    # if device:
    #    model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)            
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)


def create_chainer_supervised_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                # prepare_batch=_prepare_batch,
                                prepare_batch=chainer_prepare_batch,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):
    metrics = metrics or {}

    # if device:
    #     model.to(device)

    def _inference(engine, batch):
        model.eval()
        # with torch.no_grad():
        with chainer.no_backprop_mode():
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            y_pred = model(x)
            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
