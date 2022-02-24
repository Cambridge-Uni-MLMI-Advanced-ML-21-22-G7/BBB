import logging
from tqdm import tqdm

import torch
from torch import nn, Tensor


logger = logging.getLogger(__name__)


def train_with_tqdm(net: nn.Module, train_data: Tensor, epochs: int, eval_data: Tensor = None):
    """Wrapper for training models that will print the progress with tqdm nicely.

    This function will also save the latest model, and metrics for display in Tensorboard.

    :param net: network to be trained
    :type net: nn.Module
    :param train_data: training data
    :type train_data: Tensor
    :param epochs: number of epochs to train for
    :type epochs: int
    :param eval_data: optional evaluation data, defaults to None
    :type eval_data: Tensor, optional
    """
    with tqdm(range(epochs), unit="batch") as t_epoch:
        for epoch in t_epoch:
            # Update the tqdm toolbar
            t_epoch.set_description(f"Epoch {epoch}")

            # Run a training step
            loss = net.train(train_data)

            # Write loss to tensorboard
            net.writer.add_scalar('Training Loss', loss, epoch)

            # If you want to check the parameter values, switch log level to debug
            logger.debug(net.optimizer.param_groups)

            if eval_data is not None:
                net.eval(eval_data)

                # Write accuracy to tensorboard
                net.writer.add_scalar(f'Training {net.eval_metric}', net.eval_score, epoch)

                # Update tqdm progress bar
                t_epoch.set_postfix_str(f'Loss: {loss:.5f}, {net.eval_metric}: {net.eval_score:.5f}')
            else:
                t_epoch.set_postfix_str(f'Loss: {loss:.5f}')

            if not hasattr(net, 'best_loss') or net.best_loss is None:
                net.best_loss = loss
            
            # Save the latest model
            if loss <= net.best_loss:
                net.best_loss = loss
                torch.save(net.model.state_dict(), net.save_model_path)
