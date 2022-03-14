import logging
import argparse
from ctypes import ArgumentError

from bbb.tasks.regression import (
    run_bbb_regression_training, run_bbb_regression_evaluation, run_dnn_regression_training, run_dnn_regression_evaluation
)
from bbb.tasks.classification import (
    run_bbb_mnist_classification_training, run_bbb_mnist_classification_evaluation, run_dnn_mnist_classification_training, run_dnn_mnist_classification_evaluation
)
from bbb.tasks.bandit import run_rl_training


logger = logging.getLogger(__name__)


def init_argparse() -> argparse.ArgumentParser:
    """Parse the command line arguments passed to the bbb module.

    :return: Argument parser object
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        usage="%(prog)s [MODEL TYPE] [-d]...",
        description="Run the Bayes-by-Backprop code."
    )
    parser.add_argument(
        'model_type', choices=['reg', 'class', 'rl'],
        help='Model to run'
    )
    parser.add_argument(
        '--deterministic',
        '-d',
        action='store_true',
        help='Whether to run in deterministic (i.e., non-Bayesian) mode'
    )
    parser.add_argument(
        '--evaluate',
        '-e',
        help='Path of model to run evaluation against'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Sets the log level to DEBUG'
    )
    return parser

def main() -> None:
    """Parse the command line arguments passed and invoke the appropriate action.

    :raises ArgumentError: Raised for argument combinations not yet implemented.
    """
    logger.info('BBB started...')

    parser = init_argparse()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.model_type == 'reg':
        if args.deterministic:
            if args.evaluate:
                run_dnn_regression_evaluation(args.evaluate)
            else:
                run_dnn_regression_training()
        else:
            if args.evaluate:
                run_bbb_regression_evaluation(args.evaluate)
            else:
                run_bbb_regression_training()
    elif args.model_type == 'class':
        if args.deterministic:
            if args.evaluate:
                run_dnn_mnist_classification_evaluation(args.evaluate)
            else:
                run_dnn_mnist_classification_training()
        else:
            if args.evaluate:
                run_bbb_mnist_classification_evaluation(args.evaluate)
            else:
                run_bbb_mnist_classification_training()
    elif args.model_type == 'rl':
        run_rl_training()
    else:
        raise ArgumentError(f'Model type {args.model_type} not recognised')

    logger.info('BBB completed')

if __name__ == '__main__':
    main()
