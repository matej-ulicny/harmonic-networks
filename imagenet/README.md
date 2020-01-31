[![SotaBench](https://img.shields.io/endpoint.svg?url=https://sotabench.com/api/v0/badge/gh/matej-ulicny/harmonic-networks)](https://sotabench.com/user/matejulicny/repos/matej-ulicny/harmonic-networks)

# Harmonic Networks on ImageNet

The code used for experiments in papers [**Harmonic Networks for Image Classification**](https://bmvc2019.org/wp-content/uploads/papers/0628-paper.pdf) and [**Harmonic Convolutional Networks based on Discrete Cosine Transform**](https://arxiv.org/abs/2001.06570)

The implementation is based on the pytorch example code https://github.com/pytorch/examples/tree/master/imagenet and torchvision model definitions https://github.com/pytorch/vision/tree/master/torchvision/models

ImageNet validation set errors:

| Method | Parameters | Top-1 % | Top-5 % |
| ------ | :--------: | :-----: | :-----: |
| VGG16-BN | 138.4M | 26.33 | 8.26 |
| Harm-VGG16-BN | 138.4M | 25.55 | 8.01 |
| ResNet-50 (no pool) | 25.6M | 23.81 | 6.98 |
| Harm1-ResNet-50 | 25.6M | **22.97** | **6.48** |
| Harm-ResNet-50 | 25.6M | 23.11 | 6.63 |
| Harm-ResNet-50 (avgpool) | 25.6M | 23.1 | 6.53 |
| Harm-ResNet-50 progr.&lambda; | 19.7M | 23.12 | 6.61 | 
| ResNet-101 (maxpool) | 44.5M | 22.63 | 6.44 |
| Harm-ResNet-101 | 44.5M | **21.48** | **5.75** |

[Code for harmonic ResNext models](resnext/)

## Requirements

Tested on Python 3.5, PyTorch 1.1.
Install the required packages by running:

```
pip install -r requirements.txt
```

## Running the code

Harmonic ResNets and VGG networks implemented.
To train harmonic ResNet-50 with progressive lambda compression (None or int level per set of blocks, 4 in total) and average pooling:

```
python main.py -a resnet50 ./path/to/imagenet/ --batch-size 256 --harm_root --harm_res_blocks --levels None None 3 2 --pool avg
```

Models with increased stride can be built by simply ommiting --pool argument. Compression and pool arguments do not affect VGG networks.

Testing is performed via the same script. Pretrained are only Harm-ResNet-50 and Harm-ResNet-101, which you can test with a command specifying -a {resnet50,resnet101}:

```
python main.py -a resnet101 ./path/to/imagenet/ --batch-size 256 --harm_root --harm_res_blocks --evaluate --pretrained
```

## Cite

```
@inproceedings{Ulicny19b,
  title = {Harmonic Networks for Image Classification},
  author = {Ulicny, Matej and Krylov, Vladimir A and Dahyot, Rozenn},
  booktitle={Proceedings of the British Machine Vision Conference},
  year = {2019},
  month={Sep.}
}
```
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

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.
