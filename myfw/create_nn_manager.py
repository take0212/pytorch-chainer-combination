from myfw.mychainer.mlp_nn_manager import mlp_nn_manager as chainer_mlp_nn_manager
from myfw.mytorch.mlp_nn_manager import mlp_nn_manager as torch_mlp_nn_manager

def create_nn_manager(args):

    nn_manager = None
    if args.dataset== 'mnist':
        if args.framework_type == 'chainer':
            nn_manager = chainer_mlp_nn_manager()
        elif args.framework_type == 'pytorch':
            nn_manager = torch_mlp_nn_manager()
        else:
            raise ValueError
    else:
        raise ValueError

    # parameter
    nn_manager.set_args(args)

    return nn_manager
