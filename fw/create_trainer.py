from fw.chainer_trainer.chainer_trainer_chainer_model import chainer_trainer_chainer_model
from fw.chainer_trainer.chainer_trainer_torch_model import chainer_trainer_torch_model
from fw.torch_trainer.torch_trainer_torch_model import torch_trainer_torch_model
from fw.torch_trainer.torch_trainer_chainer_model import torch_trainer_chainer_model

def create_trainer(args):

    trainer = None
    if args.dataset== 'mnist':
        if args.trainer_framework_type == 'chainer':
            trainer = ()
            if args.model_framework_type == 'chainer':
                trainer = chainer_trainer_chainer_model()
            elif args.model_framework_type == 'pytorch':
                trainer = chainer_trainer_torch_model()
            else:
                raise ValueError
        elif args.trainer_framework_type == 'pytorch':
            if args.model_framework_type == 'pytorch':
                trainer = torch_trainer_torch_model()
            elif args.model_framework_type == 'chainer':
                trainer = torch_trainer_chainer_model()
            else:
                raise ValueError
        else:
            raise ValueError
    else:
        raise ValueError

    # parameter
    trainer.set_args(args)

    return trainer
