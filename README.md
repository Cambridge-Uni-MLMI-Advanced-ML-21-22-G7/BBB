# BBB

## Environment Setup/Installation

Create a fresh Python 3.9 environment using venv, pyenv or conda (as preferred). Then, execute `setup.sh` to install the required python libraries and versions.

For mac users, the following would be appropriate.

```
brew install python@3.9
python3.9 -m venv .venv
source .venv/bin/activate
./setup.sh
```

### M1 Torch Issue

If running on M1 you will need to ensure the following library versions are being used. These should already be set in `requirements.txt`.

```
pip install --upgrade torch==1.9.0
pip install --upgrade torchvision==0.10.0
```

## Citations

Whilst all code written is our own, the following repositories were used as sources of inspiration when implementing this project:

- [Bayesian Neural Networks](https://github.com/JavierAntoran/Bayesian-Neural-Networks)
- [Weight Uncertainty](https://github.com/danielkelshaw/WeightUncertainty)
- [PyTorch Bayesian CNN](https://github.com/kumar-shridhar/PyTorch-BayesianCNN)
- [Weight Uncertainty in Neural Networks](https://github.com/saxena-mayur/Weight-Uncertainty-in-Neural-Networks)
