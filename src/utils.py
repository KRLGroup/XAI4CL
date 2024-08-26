import argparse
import numpy as np
from omegaconf import OmegaConf
import random

import torch
import torchvision
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import SGD, Adam, RAdam
from torch.optim.lr_scheduler import StepLR

from avalanche.benchmarks.classic import *
from avalanche.benchmarks.scenarios.validation_scenario import benchmark_with_validation_stream
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics
)
from avalanche.logging import InteractiveLogger
from avalanche.models import *
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.supervised import Naive


from backward_transfer import bwt_metrics
from forward_transfer import forward_transfer_metrics
from plugins import RRRPlugin, EPRPlugin, MetricsCheckpoint

def parse_arguments():
    parser = argparse.ArgumentParser(description='XAI4CL: benchmarks for XAI-guided continual learning.')

    parser.add_argument('--config',  type=str, default='../configs/config.yaml')
    parser.add_argument('overrides', nargs='*', help="Any key = value arguments to override config values")

    flags =  parser.parse_args()
    args = OmegaConf.load(flags.config)
    return args

def set_all_seeds(seed):
    # Set seed for PyTorch
    torch.manual_seed(seed)

    # Set seed for CUDA operations (if using GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set seed for random module
    random.seed(seed)
    # Set seed for NumPy operations
    np.random.seed(seed)

def get_model(name, num_classes, benchmark):
    if name == 'mlp':
        model = SimpleMLP(num_classes=num_classes)
    elif name == 'slim-resnet18':
        #model = torchvision.models.resnet18(num_classes=num_classes)
        model = SlimResNet18(nclasses=num_classes, nf=21)
        if benchmark == 'mnist-split':
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    elif name == 'resnet18':
        model = torchvision.models.resnet18(num_classes=num_classes)
        if benchmark == 'mnist-split':
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
    elif name == 'resnet34':
        model = torchvision.models.resnet34(num_classes=num_classes)
    else:
        print('The model is not included yet: choose one from this list [mlp, resnet18, resnet34]')
    return model

def get_benchmark(name, n_tasks, seed, train_transform=None, eval_transform=None):
    if name == 'mnist-split':
        if train_transform is None and eval_transform is None:
            benchmark = SplitMNIST(n_experiences=n_tasks, shuffle=False, seed=seed)
        else:
            benchmark = SplitMNIST(n_experiences=n_tasks, shuffle=False, seed=seed, train_transform=train_transform, eval_transform=eval_transform)
    elif name == 'cifar10-split':
        if train_transform is None and eval_transform is None:
            benchmark = SplitCIFAR10(n_experiences=n_tasks, shuffle=False, seed=seed)
        else:
            benchmark = SplitCIFAR10(n_experiences=n_tasks, shuffle=False, seed=seed, train_transform=train_transform, eval_transform=eval_transform)
    elif name == 'cifar100-split':
        if train_transform is None and eval_transform is None:
            benchmark = SplitCIFAR100(n_experiences=n_tasks, shuffle=False, seed=seed)
        else:
            benchmark = SplitCIFAR100(n_experiences=n_tasks, shuffle=False, seed=seed, train_transform=train_transform, eval_transform=eval_transform)
    elif name == 'cub200-split':
        if train_transform is None and eval_transform is None:
            benchmark = SplitCUB200(n_experiences=n_tasks, shuffle=False, seed=seed)
        else:
            benchmark = SplitCUB200(n_experiences=n_tasks, shuffle=False, seed=seed, train_transform=train_transform, eval_transform=eval_transform)
    else:
        print('The benchmark is not included yet: choose one from this list [split-mnist, cifar10-split, cifar100-split]')

    # Ensure that the split is reproducible
    torch.manual_seed(seed)
    random.seed(seed)
    benchmark = benchmark_with_validation_stream(benchmark, validation_size=0.2, shuffle=True)
    return benchmark

def get_optimizer(args, model):
    assert args.train.optimizer in ['sgd', 'adam', 'radam'], "Optimizer not supported. Supported ['sgd', 'adam','radam']."
    
    if args.train.optimizer == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.train.lr, momentum=args.train.momentum, weight_decay=args.train.weight_decay)
    elif args.train.optimizer == 'radam':
        optimizer = RAdam(model.parameters(), lr=args.train.lr)
    elif args.train.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.train.lr)
    
    if args.experiment.strategy == 'rrr':
        assert args.strategy.xai_optimizer == 'adam', 'XAI optimizer not supported. Supported [adam].'
        if args.strategy.xai_optimizer == 'adam':
            xai_optimizer = Adam(model.parameters(), lr=args.strategy.xai_lr)
    else:
        xai_optimizer = None
    
    return optimizer, xai_optimizer

def get_loss(args):
    assert args.train.loss in ['cross_entropy'], "Loss not supported. Supported ['cross_entropy']."
    
    if args.train.loss == 'cross_entropy':
        loss = CrossEntropyLoss()
    if args.experiment.strategy == 'rrr':
        assert args.strategy.xai_loss == 'l1', 'XAI loss not supported. Supported [l1].'
        if args.strategy.xai_loss == 'l1':
            xai_loss = L1Loss()
    else:
        xai_loss = None
    
    return loss, xai_loss

def get_strategy(args, model, checkpoint_dir, device):
    optimizer, xai_optimizer = get_optimizer(args, model)
    loss, xai_loss = get_loss(args)
    
    '''
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True),
        loss_metrics(epoch=True, experience=True),
        bwt_metrics(experience=True),
        forward_transfer_metrics(experience=True)
    )
    '''

    interactive_logger = InteractiveLogger()
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        bwt_metrics(experience=True),
        forward_transfer_metrics(experience=True),
        loggers=[interactive_logger],
    )

    strategy = Naive(model, optimizer, loss,
                    train_mb_size=args.train.batch_size, train_epochs=args.train.epochs,
                    eval_mb_size=args.train.batch_size, eval_every=1, evaluator=eval_plugin, 
                    device=device)
    
    strategy_plugins = []
    if args.experiment.strategy == 'ewc':
        strategy_plugins.append(EWCPlugin(ewc_lambda=args.strategy.ewc_lambda))
    elif args.experiment.strategy == 'mas':
        strategy_plugins.append(MASPlugin(lambda_reg=args.strategy.lambda_reg,
                                          alpha=args.strategy.alpha))
    elif args.experiment.strategy == 'er':
        strategy.mem_size = args.strategy.mem_size
        strategy_plugins.append(OurReplayPlugin(batch_size=args.train.batch_size,
                                                batch_size_mem=args.strategy.batch_size_mem, 
                                                task_balanced_dataloader=True))
    elif args.experiment.strategy == 'rrr':
        strategy.plugins += [RRRPlugin(xai_loss=xai_loss,
                                        xai_optimizer=xai_optimizer,
                                        xai_regularizer=args.strategy.xai_regularizer,
                                        mem_size=args.strategy.mem_size,
                                        mem_adaptive_size=args.strategy.mem_adaptive_size,
                                        num_experiences=args.experiment.n_tasks,
                                        batch_size=args.train.batch_size,
                                        batch_size_mem=args.strategy.batch_size_mem, 
                                        task_balanced_dataloader=True,
                                        )]
    elif args.experiment.strategy == 'epr':
        strategy.plugins += [EPRPlugin(mem_size=args.strategy.mem_size,
                                        mem_adaptive_size=args.strategy.mem_adaptive_size,
                                        num_experiences=args.experiment.n_tasks,
                                        batch_size=args.train.batch_size,
                                        batch_size_mem=args.strategy.batch_size_mem, 
                                        task_balanced_dataloader=True)]
        
    if args.experiment.benchmark == 'cifar100-split':
        scheduler = StepLR(optimizer, step_size=args.train.epochs//3, gamma=0.3)
        scheduler_plugin = LRSchedulerPlugin(scheduler, step_granularity="epoch", first_exp_only=False)
        strategy_plugins.append(scheduler_plugin)
        
    strategy.plugins += strategy_plugins
    strategy.plugins += [MetricsCheckpoint(n_tasks=args.experiment.n_tasks,
                                           checkpoint_every=args.setup.checkpoint_every,
                                           checkpoint_dir=checkpoint_dir,
                                           verbose=True)]

    return strategy