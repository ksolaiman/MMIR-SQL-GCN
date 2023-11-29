"""SimGNN runner."""

from utils import tab_printer
from simgnn import SimGNNTrainer
from param_parser import parameter_parser

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    tab_printer(args)
    trainer = SimGNNTrainer(args)
    if not args.evaluate_only:
        trainer.fit()
    else:
        trainer.score()

if __name__ == "__main__":
    main()
