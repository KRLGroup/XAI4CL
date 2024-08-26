import os
import torch

from utils import *

def training_loop(cl_strategy, train_stream, valid_stream):
  for exp_id, train_exp in enumerate(train_stream):
      print("Start of experience: ", train_exp.current_experience)
      print("Current Classes: ", train_exp.classes_in_this_experience)

      # train returns a dictionary which contains all the metric values
      _ = cl_strategy.train(train_exp, eval_streams=[valid_stream.exps_iter])
      print('Trained experience ', train_exp.current_experience)

def main():
    # Parse arguments
    args = parse_arguments()

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set seed for reproducibility
    set_all_seeds(args.experiment.seed)

    # Define model
    model = get_model(args.experiment.model, num_classes=args.experiment.n_classes, benchmark=args.experiment.benchmark)
    model = model.to(device)

    # Define CL Benchmark
    benchmark = get_benchmark(args.experiment.benchmark, args.experiment.n_tasks, seed=args.experiment.seed)
    train_stream = benchmark.train_stream
    valid_stream = benchmark.valid_stream
    test_stream = benchmark.test_stream

    # Create checkpoint directory
    checkpoint_dir = args.setup.checkpoint_dir+'/'+args.experiment.benchmark+'_'+args.experiment.strategy
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    # Define continual learning strategy
    cl_strategy = get_strategy(args, model, checkpoint_dir, device)

    # Training loop
    training_loop(cl_strategy, train_stream, valid_stream)


if __name__ == "__main__":
    main()