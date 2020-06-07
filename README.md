# PyTorch-Chainer-combination

I combined PyTorch or Chainer trainer and PyTorch or Chainer model to practice the migration.

## Samples

There are 4 samples.

- Run PyTorch trainer with PyTorch model (typical)
- Run PyTorch trainer with Chainer model (combination)
- Run Chainer trainer with Chainer model (typical)
- Run Chainer trainer with PyTorch model (combination)

## Execution procedure

1.Select framework for trainer and model.

Edit exec_trainer.py.

```
args.trainer_framework_type = 'pytorch'
# args.trainer_framework_type = 'chainer'

args.model_framework_type = 'pytorch'
# args.model_framework_type = 'chainer'
```

2.Run exec_trainer.py.

## Acknowledgments

I would like to thank Emcastillo of Preferred Networks, Inc.
