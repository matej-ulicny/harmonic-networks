# Harmonic Networks for Holistically-Nested Edge Detection

Our harmonic HED models trained on BSDS500 are reported in [**Harmonic Convolutional Networks based on Discrete Cosine Transform**](https://arxiv.org/abs/2001.06570v1)

Models are implemented within our extension of this repository https://github.com/meteorshowers/hed-pytorch

Changelog:
* Main script allows training from sratch as well as from pretrained weights
* Models support harmonic blocks

F-scores on BSDS500 test set:

| Method | Params | Baseline ODS/OIS | Harmonic ODS/OIS |
| ------ | :----: | :--------------: | :--------------: |
| HED small | 0.1M | 0.738 / 0.756 | 0.743 / 0.763 |
| HED | 14.6M | 0.761 / 0.778 | 0.770 / 0.789 |
| HED pretrained | 14.6M | 0.777 / 0.796 | 0.782 / 0.803 |

## Requirements

Tested on Python 3.5, PyTorch 1.1.0
Install the required packages by running:

## Running the code

Download and extract the augmented dataset http://vcl.ucsd.edu/hed/HED-BSDS.tar into *data/* folder.
Default model is the full-size HED. Following arguments can be used:
* `--small` will train small version of HED
* `--harmonic` changes all conv layers to harmonic blocks
* `--pretrained` uses imagenet trained weights to initialize the model. Only full-size model is pretrained. If used together with --harmonic, download [converted VGG16 weights](https://github.com/matej-ulicny/harmonic-networks/releases/download/0.1.0/harm_vgg16_conv.pth) and place them in the same directory as the training script
* `--use-cfg` will override --lr, --maxepoch and --stepsize parameters with default ones for each model depending on the use of flags above

Example of training Harm-HED from pretrained weights:
```
python train_hed.py --output output/model_name --harmonic --pretrained --use-cfg
```
Or small Harm-HED model:
```
python train_hed.py --output output/model_name --harmonic --small --use-cfg
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
