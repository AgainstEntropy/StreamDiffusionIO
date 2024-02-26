# StreamDiiffusionIO

StreamDiiffusionIO's pipeline design is based on [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion), but especially allows using different text prompt on different samples in the denoising batch. 

A natural application of StreamDiiffusionIO is to render text streams into image streams, as in [kanji](https://github.com/AgainstEntropy/kanji).

## Installation

### Create the Env

```shell
conda create -n StreamDiffusionIO python=3.10
conda activate StreamDiffusionIO
```

### Install StreamDiffusionIO

#### For Users

```shell
pip install -i https://test.pypi.org/simple/ StreamDiffusionIO
```

#### For Developers

```shell
git clone https://github.com/AgainstEntropy/StreamDiffusionIO.git
cd StreamDiffusionIO
pip install --editable .
```

### (Optional) Accelaration with `xformers`

```shell
# For user
pip install -i https://test.pypi.org/simple/ StreamDiffusionIO[xformers]

# For dev
pip install -e '.[xformers]'
```

## Quick Start

StreamDiffusionIO is very similar to StreamDiffusion, but even more lightweight. One can use the pipeline with only a few lines of codes.

```python
import torch
from StreamDiffusionIO import LatentConsistencyModelStreamIO

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id_or_path = "runwayml/stable-diffusion-v1-5"
lora_path = "/path/to/lora/pytorch_lora_weights.safetensors"
lcm_lora_path = "/path/to/lcm-lora/pytorch_lora_weights.safetensors"

stream = LatentConsistencyModelStreamIO(
    model_id_or_path=model_id_or_path,
    lcm_lora_path=lcm_lora_path,
    lora_dict={lora_path: 1},
    resolution=128,
    device=device,
)

text = "Today I saw a beautiful sunset and it made me feel so happy."
prompt_list = text.split()

# to simulate a text stream
for prompt in prompt_list:
    image = stream(prompt)  # stream returns None during warmup
    if image is not None:
        display(image)
```

Please follow the Jupyter notebooks in [examples](./examples/) to see details.


## Acknowledgements & References

- [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion)
- [Latent Consistency Models](https://github.com/huggingface/diffusers/tree/main/examples/consistency_distillation)