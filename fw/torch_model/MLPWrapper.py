from fw.torch_model.MLP import MLP

class MLPWrapper(MLP):

    def forward(self, *args, **kwargs):
        return super(MLPWrapper, self).forward(args[0])

    def namedlinks(self, skipself=False):
        # Hack for the evaluator extension to work
        return []
