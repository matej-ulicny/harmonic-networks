[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harmonic-networks-with-limited-training/image-classification-on-stl-10)](https://paperswithcode.com/sota/image-classification-on-stl-10?p=harmonic-networks-with-limited-training)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harmonic-convolutional-networks-based-on/image-classification-on-imagenet)](https://paperswithcode.com/sota/image-classification-on-imagenet?p=harmonic-convolutional-networks-based-on)
[![SotaBench](https://img.shields.io/endpoint.svg?url=https://sotabench.com/api/v0/badge/gh/matej-ulicny/harmonic-networks)](https://sotabench.com/user/matejulicny/repos/matej-ulicny/harmonic-networks)
# Harmonic Networks

Repository implements harmonic blocks presented in following papers:
* [**Harmonic Networks for Image Classification**](https://bmvc2019.org/wp-content/uploads/papers/0628-paper.pdf)
* [**Harmonic Networks with Limited Training Samples**](https://ieeexplore.ieee.org/abstract/document/8902831)
* [**Harmonic Convolutional Networks based on Discrete Cosine Transform**](https://arxiv.org/abs/2001.06570)

Convolutional neural networks (CNNs) learn filters in order to capture local correlation patterns in feature space. In this paper we propose to revert to learning combinations of preset spectral filters by switching to CNNs with harmonic blocks. We rely on the use of the Discrete Cosine Transform (DCT) filters which have excellent energy compaction properties and are widely used for image compression. The proposed harmonic blocks rely on DCT-modeling and replace conventional convolutional layers to produce partially or fully harmonic versions of new or existing CNN architectures. We demonstrate how the harmonic networks can be efficiently compressed in a straightforward manner by truncating high-frequency information in harmonic blocks which is possible due to the redundancies in the spectral domain. We report extensive experimental validation demonstrating the benefits of the introduction of harmonic blocks into state-of-the-art CNN models in image classification, segmentation and edge detection applications.

## Requirements

All code compatible with Python 3.5+ and PyTorch 1.0+
Install the required packages by running:
```
pip install -r requirements.txt
```

## Installation

First build the package prior to running any code:
```
python setup.py develop
```

Repository implements experiments on several datasets:
* [Image classification on CIFAR10/100](cifar/)
* [Image classification on STL10](stl/)
* [Image classification on ImageNet](imagenet/)
* [Object tedetction on MS COCO and PASCAL VOC](mmdetection/)
* [Boundary prediction on BSDS500](bsds/)

## Citation
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
