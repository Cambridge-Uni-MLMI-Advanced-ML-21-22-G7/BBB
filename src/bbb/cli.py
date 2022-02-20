import logging
import argparse
from ctypes import ArgumentError

from bbb.regression import run_dnn_regression
from bbb.classify import run_bbb_mnist_classification, run_cnn_mnist_classification


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
        "-v", "--version", action="version",
        version = f"{parser.prog} version 1.0.0"
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
    return parser

def main() -> None:
    """Parse the command line arguments passed and invoke the appropriate action.

    :raises ArgumentError: Raised for argument combinations not yet implemented.
    """
    logger.info('BBB started...')

    parser = init_argparse()
    args = parser.parse_args()

    if args.model_type == 'reg':
        if args.deterministic:
            run_dnn_regression()
        else:
            raise ArgumentError('Bayesian regression not yet implemented')
    elif args.model_type == 'class':
        if args.deterministic:
            run_cnn_mnist_classification()
        else:
            run_bbb_mnist_classification()
    elif args.model_type == 'rl':
        raise ArgumentError('Reinforcement learning not yet implemented')
    else:
        raise ArgumentError(f'Model type {args.model_type} not recognised')

    logger.info('BBB completed')

if __name__ == '__main__':
    main()
