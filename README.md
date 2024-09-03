# Phils Dreambooth

## FLUX
tuto: https://github.com/bghira/SimpleTuner/blob/main/documentation/quickstart/FLUX.md
```bash
apt -y install nvidia-cuda-toolkit libgl1-mesa-glx # for runpod
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.11 you can just also use the 'python' command here.
python3.11 -m venv .venv

source .venv/bin/activate

pip install -U poetry pip
poetry install
```

CONFIG_BACKEND=json ./train.sh