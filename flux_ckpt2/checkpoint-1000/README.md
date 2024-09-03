---
license: other
base_model: "black-forest-labs/FLUX.1-dev"
tags:
  - flux
  - flux-diffusers
  - text-to-image
  - diffusers
  - simpletuner
  - not-for-all-audiences
  - lora
  - template:sd-lora
  - standard
inference: true
widget:
- text: 'unconditional (blank prompt)'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_0_0.png
- text: 'a breathtaking anime-style portrait of Philippe, capturing his essence with vibrant colors and expressive features'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_1_0.png
- text: 'a high-quality, detailed photograph of Philippe as a sous-chef, immersed in the art of culinary creation'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_2_0.png
- text: 'a lifelike and intimate portrait of Philippe, showcasing his unique personality and charm'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_3_0.png
- text: 'a cinematic, visually stunning photo of Philippe, emphasizing his dramatic and captivating presence'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_4_0.png
- text: 'an elegant and timeless portrait of Philippe, exuding grace and sophistication'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_5_0.png
- text: 'a dynamic and adventurous photo of Philippe, captured in an exciting, action-filled moment'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_6_0.png
- text: 'a mysterious and enigmatic portrait of Philippe, shrouded in shadows and intrigue'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_7_0.png
- text: 'a vintage-style portrait of Philippe, evoking the charm and nostalgia of a bygone era'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_8_0.png
- text: 'an artistic and abstract representation of Philippe, blending creativity with visual storytelling'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_9_0.png
- text: 'a futuristic and cutting-edge portrayal of Philippe, set against a backdrop of advanced technology'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_10_0.png
- text: 'A picture of Philippe'
  parameters:
    negative_prompt: 'blurry, cropped, ugly'
  output:
    url: ./assets/image_11_0.png
---

# simpletuner-lora

This is a standard PEFT LoRA derived from [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).


The main validation prompt used during training was:



```
A picture of Philippe
```

## Validation settings
- CFG: `3.0`
- CFG Rescale: `0.0`
- Steps: `20`
- Sampler: `None`
- Seed: `42`
- Resolution: `1024x1024`

Note: The validation settings are not necessarily the same as the [training settings](#training-settings).

You can find some example images in the following gallery:


<Gallery />

The text encoder **was not** trained.
You may reuse the base model text encoder for inference.


## Training settings

- Training epochs: 90
- Training steps: 1000
- Learning rate: 0.0001
- Effective batch size: 1
  - Micro-batch size: 1
  - Gradient accumulation steps: 1
  - Number of GPUs: 1
- Prediction type: flow-matching
- Rescaled betas zero SNR: False
- Optimizer: adamw_bf16
- Precision: Pure BF16
- Quantised: No
- Xformers: Not used
- LoRA Rank: 16
- LoRA Alpha: None
- LoRA Dropout: 0.1
- LoRA initialisation style: default
    

## Datasets

### dreambooth-subject
- Repeats: 0
- Total number of images: 11
- Total number of aspect buckets: 1
- Resolution: 1.048576 megapixels
- Cropped: False
- Crop style: None
- Crop aspect: None


## Inference


```python
import torch
from diffusers import DiffusionPipeline

model_id = 'black-forest-labs/FLUX.1-dev'
adapter_id = 'PhilSad/simpletuner-lora'
pipeline = DiffusionPipeline.from_pretrained(model_id)
pipeline.load_lora_weights(adapter_id)

prompt = "A picture of Philippe"

pipeline.to('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
image = pipeline(
    prompt=prompt,
    num_inference_steps=20,
    generator=torch.Generator(device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu').manual_seed(1641421826),
    width=1024,
    height=1024,
    guidance_scale=3.0,
).images[0]
image.save("output.png", format="PNG")
```

