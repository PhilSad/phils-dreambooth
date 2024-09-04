# Phils Dreambooth

## FLUX
### train
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

`CONFIG_BACKEND=json ./train.sh`

### Comfyui Infer
```bash
cd ComfyUI/models
huggingface-cli download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir=unet
huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir=vae
wget -P clip/ https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
wget -P clip/ https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors
```
