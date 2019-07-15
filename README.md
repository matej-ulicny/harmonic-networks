# Harmonic Networks

The code used for experiments in papers [**Harmonic Networks for Image Classification**](https://www.scss.tcd.ie/Rozenn.Dahyot/pdf/Harmonic_BMVC2019.pdf) and [**Harmonic Networks with Limited Training Samples**](https://arxiv.org/abs/1905.00135)

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
and
```
@inproceedings{Ulicny19,
  author = {Ulicny, Matej and Krylov, Vladimir A and Dahyot, Rozenn},
  booktitle={European Signal Processing Conference (EUSIPCO)},
  title = {Harmonic Networks with Limited Training Samples},
  year={2019}, 
  month={Sep.},
  url={https://arxiv.org/abs/1905.00135}
}
```

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.


