import logging

import torch


logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    logger.debug('CUDA available - PyTorch will use GPU')
    dev = 'cuda:0'
else:
    logger.debug('CUDA unavailable - PyTorch will use CPU')
    dev = 'cpu'
DEVICE = torch.device(dev)
