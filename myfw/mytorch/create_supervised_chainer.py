# ignite.engine.__init__.py

# import torch
# from ignite.engine.engine import Engine, State, Events
from ignite.engine.engine import Engine
# from ignite.utils import convert_tensor
# from ignite.engine import _prepare_batch
# ----
import chainer

def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.

    """
    x, y = batch
    # return (convert_tensor(x, device=device, non_blocking=non_blocking),
    #         convert_tensor(y, device=device, non_blocking=non_blocking))
    x = chainer.Variable(x.numpy())
    y = chainer.Variable(y.numpy())
    x.to_device(device)
    y.to_device(device)
    return (x, y)

def create_supervised_chainer_trainer(model, optimizer, loss_fn,
                              device=None, non_blocking=False,
                              prepare_batch=_prepare_batch,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    # if device:
    #    model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)            

        y_pred = model(x)

        # no good
        y_pred = chainer.Variable(y_pred.cpu().numpy())
        y_pred.to_device(device)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return Engine(_update)

def create_supervised_chainer_evaluator(model, metrics=None,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
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

            # no good
            y_pred = chainer.Variable(y_pred.cpu().numpy())
            y_pred.to_device(device)

            return output_transform(x, y, y_pred)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine
