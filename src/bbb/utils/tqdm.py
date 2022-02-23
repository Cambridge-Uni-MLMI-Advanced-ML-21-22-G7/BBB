import logging
from tqdm import tqdm

import torch
from torch import nn, Tensor


logger = logging.getLogger(__name__)


def train_with_tqdm(net: nn.Module, train_data: Tensor, epochs: int, eval_data: Tensor = None):
    """Wrapper for training models that will print the progress with tqdm nicely.

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
            t_epoch.set_description(f"Epoch {epoch}")
            loss = net.train(train_data)

            # If you want to check the parameter values, switch log level to debug
            logger.debug(net.optimizer.param_groups)

            if eval_data is not None:
                net.eval(eval_data)
                t_epoch.set_postfix_str(f'Loss: {loss:.5f}; Acc: {net.acc:.5f}')

                if not hasattr(net, 'best_acc') or net.best_acc is None:
                    net.best_acc = net.acc
                
                if net.acc >= net.best_acc:
                    net.best_acc = net.acc
                    torch.save(net.model.state_dict(), net.save_model_path)
            else:
                t_epoch.set_postfix_str(f'Loss: {loss:.5f}')
