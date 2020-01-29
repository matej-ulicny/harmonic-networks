[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harmonic-networks-with-limited-training/image-classification-on-stl-10)](https://paperswithcode.com/sota/image-classification-on-stl-10?p=harmonic-networks-with-limited-training)
[![SotaBench](https://img.shields.io/endpoint.svg?url=https://sotabench.com/api/v0/badge/gh/matej-ulicny/harmonic-networks)](https://sotabench.com/user/matejulicny/repos/matej-ulicny/harmonic-networks)
# Harmonic Networks

The code used for experiments in papers [**Harmonic Networks for Image Classification**](https://bmvc2019.org/wp-content/uploads/papers/0628-paper.pdf), [**Harmonic Networks with Limited Training Samples**](https://ieeexplore.ieee.org/abstract/document/8902831) and [**Harmonic Convolutional Networks based on Discrete Cosine Transform**](https://arxiv.org/abs/2001.06570)

Convolutional neural networks (CNNs) are very popular nowadays for image processing. CNNs allow one to learn optimal filters in a (mostly) supervised machine learning context. However this typically requires abundant labelled training data to estimate the filter parameters. Alternative strategies have been deployed for reducing the number of parameters and / or filters to be learned and thus decrease overfitting. In the context of reverting to preset filters, we propose here a computationally efficient harmonic block that uses Discrete Cosine Transform (DCT) filters in CNNs. In this work we examine the performance of harmonic networks in limited training data scenario. We validate experimentally that its performance compares well against scattering networks that use wavelets as preset filters.

The implementation is based on the original PyTorch WRN code from https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch

Test errors in % (median of 5 runs) on CIFAR datasets:

| Method | dropout | compression | parameters | CIFAR-10 | CIFAR-100 |
| ------ | :-----: | :---------: | :--------: | :------: | :-------: |
| WRN-28-10 | &#10004; | | 36.5M | 3.91 | 18.75 |
| Harm1-WRN-28-10 | | | 36.5M | 3.90 | 18.80 |
| Harm1-WRN-28-10 | &#10004; | | 36.5M | **3.64** | **18.57** |
| Harm-WRN-28-10 | &#10004; | | 36.5M | 3.86 | **18.57** |
| Harm-WRN-28-10 | &#10004; | &lambda;=3 | 24.4M | 3.84 | 18.58 |
| Harm-WRN-28-10 | &#10004; | &lambda;=2 | 12.3M | 4.25 | 19.97 |
| WRN-28-8 | &#10004; | | 23.4M | 4.01 | 19.38 |
| WRN-28-6 | &#10004; | | 13.1M | 4.09 | 20.17 |

Classification accuracy in % (mean &plusmn; std) on STL test set using folds or all 5000 training samples:

| Method | 10-folds | all |
| ------ | -------- | --- |
| WRN 16-8 | 73.50 &plusmn; 0.87 | 87.29 &plusmn; 0.21 |
| Scat + WRN | 76.00 &plusmn; 0.60 | 87.60 |
| Harm WRN 16-8 | 76.95 &plusmn; 0.93 | **90.45 &plusmn; 0.12** |
| Harm WRN 16-8 &lambda;=3 | 76.65 &plusmn; 0.90 | 90.39 &plusmn; 0.08 |
| Harm WRN 16-8 progressive &lambda; | **77.19 &plusmn; 1.02** | 90.28 &plusmn; 0.20 |

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

## Requirements

Tested on Python 2.7, 3.5, PyTorch 4.1, 1.0, 1.1.
Install the required packages by running:

```
pip install -r requirements.txt
```

## Running the code

To run the code, --dataset and WRN parameters need to be specified such as --depth N, --width K, optional dropout --dropout 0.X or nesterov momentum --nesterov. By default fully harmonic WRN is trained, which can be changed by importing one of the other network definitions from models folder. Parameter &lambda; from the papers can be set to 2 or 3 specifying --level. Model with progressive compression has its own definition.

### Running on CIFAR

For training harmonic WRN-28-10 with lambda compression of level 3 and dropout of 0.3 with 2 GPUs on CIFAR100 dataset run:

```
python main.py --save ./save_location --depth 28 --width 10 --dropout 0.3 --level 3 --dataset CIFAR100 --epochs 200 --epoch_step [60,120,160] --ngpu 2 --gpu_id 0,1
```

To train a network on one of the predefined limited subsets --subset_size (one of 100, 500, 1000) and --subset_id (1-5) have to be specified. Number of epochs and scheduler steps have to be adjusted accordingly. An example of using subset number 2 with 500 samples:

```
python main.py --save ./save_location --depth 28 --width 10 --dropout 0.2 --epochs 200 --epoch_step [1200,2400,3200] --subset_size 4000 --subset_id 2 --ngpu 2 --gpu_id 0,1
```

### Running on STL

For training harmonic WRNs on STL dataset is analogous to running on CIFAR:

```
python main_stl.py --save ./save_location --depth 16 --width 8 --dropout 0.3 --batch_size 32 --ngpu 2 --gpu_id 0,1
```

By default the whole set of 5000 images is used for training. Training on pre-defined fold is possible by specifying --fold parameter (0-9). Number of epochs and scheduler steps are adjusted automatically. Training on fold number 4:

```
python main_stl.py --save ./save_location --depth 16 --width 8 --dropout 0.3  --batch_size 32 --fold 4 --ngpu 2 --gpu_id 0,1
```

### Running on ImageNet

Harmonic ResNets and VGG networks implemented.
To train harmonic ResNet-50 with progressive lambda compression (None or int level per set of blocks, 4 in total) and average pooling:

```
python main.py -a resnet50 ./path/to/imagenet/ --batch-size 256 --harm_root --harm_res_blocks --levels None None 3 2 --pool avg
```

Models with increased stride can be built by simply ommiting --pool argument. Compression and pool arguments do not affect VGG networks.

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
@inproceedings{Ulicny19,
  author = {Ulicny, Matej and Krylov, Vladimir A and Dahyot, Rozenn},
  booktitle={27th European Signal Processing Conference (EUSIPCO)},
  title = {Harmonic Networks with Limited Training Samples},
  doi={10.23919/EUSIPCO.2019.8902831},
  ISSN={2219-5491},
  year={2019}, 
  month={Sep.},
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
