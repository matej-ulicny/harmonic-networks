# Harmonic ResNext Networks

Our harmonic ResNeXt models trained on ImageNet-1k are reported in [**Harmonic Convolutional Networks based on Discrete Cosine Transform**](https://arxiv.org/abs/2001.06570)

Models are implemented within our extension of this repository https://github.com/rwightman/pytorch-image-models of version 0.1.8

Changelog:
* Random rotation augmentation
* Stochastic depth implementation from https://github.com/lukemelas/EfficientNet-PyTorch
* Harmonic blocks inside the model

ImageNet validation set errors:

| Method | Params | Top-1 (224) | Top-5 (224) | Top-1 (320) | Top-5 (320) | Weights |
| ------ | :--------: | :---------: | :---------: | :---------: | :---------: | :--------: |
| SE-ResNext-101 32x4d | 49.0M | 19.81 | 4.90 | 18.80 | 4.19 | [download](https://drive.google.com/file/d/1_tnT2ui5KSmHnARRaqgiyfbWRwbpJxXr/view?usp=sharing) |
| Harm-SE-ResNeXt-101 32x4d | 49.0M | 19.55 | 4.79 | 18.72 | 4.23 | [download](https://drive.google.com/file/d/1LtHPMb9IqJ2tWUhy_WnCXwq3dtJDPgbX/view?usp=sharing) |
| Harm-SE-ResNeXt-101 64x4d | 88.2M | **18.36** | **4.37** | **17.34** | **3.71** | [download](https://drive.google.com/file/d/1LtHPMb9IqJ2tWUhy_WnCXwq3dtJDPgbX/view?usp=sharing) |

## Requirements

Tested on Python 3.5, PyTorch 1.1.0
Install the required packages by running:

```
pip install -r requirements.txt
```

## Testing pretrained models

To test harmonic ResNeXt-101 64x4d using 2 gpus and image crop 320x320:

```
python validate.py ./path/to/imagenet/validation --model harm_se_resnext101_64x4d -b 256 --checkpoint ./harm_se_resnext101_64x4d.pth.tar --num-gpu 2 --img-size 320
```

## Cite

```
@misc{Ulicny20,
    title={Harmonic Convolutional Networks based on Discrete Cosine Transform},
    author={Matej Ulicny and Vladimir A. Krylov and Rozenn Dahyot},
    year={2020},
    eprint={2001.06570},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
