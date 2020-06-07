import argparse
# ----
from fw.create_trainer import create_trainer

def set_custom_params(args):

    # args.trainer_framework_type = 'chainer'
    args.trainer_framework_type = 'pytorch'
    
    args.model_framework_type = 'chainer'
    # args.model_framework_type = 'pytorch'

    # args.max_epochs = 20
    args.max_epochs = 3

    # args.log_interval = 5
    args.log_interval = 1

    # args.eval_interval = 5
    args.eval_interval = 1

    args.lazy = True

    pass

def set_params():

    # parameter
    parser = argparse.ArgumentParser(description='example')
    parser.add_argument("--trainer-framework-type", type=str, default='pytorch', help='framework type')
    parser.add_argument("--model-framework-type", type=str, default='pytorch', help='model framework type')
    parser.add_argument("--dataset", type=str, default='mnist', help='dataset')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--max-epochs', type=int, default=20, help='max epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='mini-batch size')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--log-interval', type=int, default=5, help='log interval')
    parser.add_argument('--eval-interval', type=int, default=5, help='eval interval')
    parser.add_argument('--retain-num', type=int, default=1, help='retain num')
    parser.add_argument('--resume_filename', type=str, help='resume filename')
    parser.add_argument('--call-before-training', action='store_true', help='call before training')
    parser.add_argument('--lazy', action='store_true', help='lazy')
    args = parser.parse_args()
   
    set_custom_params(args)

    return args

def main():
    # parameter
    args = set_params()

    # trainer
    trainer = create_trainer(args)

    # train
    trainer.train()

if __name__ == '__main__':
    main()
