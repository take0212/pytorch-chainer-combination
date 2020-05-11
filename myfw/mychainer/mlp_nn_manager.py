from chainer.datasets import mnist
# ----
from myfw.mychainer.fw_nn_manager import fw_nn_manager

class mlp_nn_manager(fw_nn_manager):

    # Dataset
    def set_dataset(self):

        if self.dataset == 'mnist':
            self.train_dataset, self.valid_dataset = mnist.get_mnist()
        else:
            raise ValueError

    # model
    def set_model(self):
        if self.model_framework_type == 'chainer':

            import chainer.links as L
            from myfw.mychainer.model.MLP import MLP
            model = MLP()
            self.model = L.Classifier(model)

        elif self.model_framework_type == 'pytorch':

            from myfw.mypytorch.model.MLP import MLP
            self.model = MLP()

        else:
            raise ValueError

        super(mlp_nn_manager, self).set_model()