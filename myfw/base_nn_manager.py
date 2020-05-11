import os

class base_nn_manager():
    def __init__(self):

        self.framework_type = None
        self.model_framework_type = None
        self.dataset = None
        self.device = None
        self.max_epochs = None
        self.batch_size = None
        self.seed = None
        self.lr = None
        self.momentum = None
        self.log_interval = None
        self.eval_interval = None
        self.retain_num = None
        self.resume_filename = None
        self.call_before_training = None
        
        self.resume_filepath = None

        # ----

        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

        self.model = None  
        self.optimizer = None
        self.trainer = None

        self.input_size = None

    def set_args(self, args):
        for name, val in vars(args).items():
            setattr(self, name, val)

    def set_param(self):
        if self.resume_filename is not None:
            self.resume_filepath  = os.path.join(self.out_model, self.resume_filename)

    # Dataset
    def set_dataset(self):
        raise NotImplementedError

    # DataLoader, Iterator
    def set_dataloader(self):
        raise NotImplementedError

    # model
    def set_model(self):
        raise NotImplementedError

    # Optimizer
    def set_optimizer(self):
        raise NotImplementedError

    # Ignite, Updater
    def set_trainer(self):
        raise NotImplementedError

    # event handler
    def set_event_handler(self):
        raise NotImplementedError

    # resume
    def resume(self):
        raise NotImplementedError

    # run
    def run(self):
        raise NotImplementedError

    def train(self):
        self.set_param()
        self.set_dataset()
        self.set_dataloader()
        self.set_model()
        self.set_optimizer()
        self.set_trainer()
        self.set_event_handler()
        self.resume()
        self.run()
