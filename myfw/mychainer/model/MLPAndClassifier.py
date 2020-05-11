# import
import chainer
import chainer.links as L
import chainer.functions as F

from myfw.mychainer.model.MLP import MLP

class MLPAndClassifier(MLP):

    def forward(self, x):
        y = super(MLPAndClassifier, self).forward(x)
        return F.log_softmax(y)
