# BBB

# Environment Setup/Installation

Create a fresh Python 3.9 environment using venv, pyenv or conda (as preferred). Then, execute `setup.sh` to install the required python libraries and versions.

For mac users, the following would be appropriate.

```
brew install python@3.9
python3.9 -m venv .venv
source .venv/bin/activate
./setup.sh
```

# M1 Torch Issue

If running on M1, do this:
```
pip install --upgrade torch==1.9.0
pip install --upgrade torchvision==0.10.0
```
