def create_nn_manager(args):

    nn_manager = None
    if args.dataset== 'mnist':
        if args.framework_type == 'chainer':
            from myfw.mychainer.mlp_nn_manager import mlp_nn_manager
            nn_manager = mlp_nn_manager()
        elif args.framework_type == 'pytorch':
            from myfw.mypytorch.mlp_nn_manager import mlp_nn_manager
            nn_manager = mlp_nn_manager()
        else:
            raise ValueError
    else:
        raise ValueError

    # parameter
    nn_manager.set_args(args)

    return nn_manager
