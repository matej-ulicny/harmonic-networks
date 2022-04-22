# Harmonic Wide Residual Networks on CIFAR10/100

The code used for experiments in papers [**Harmonic Networks for Image Classification**](https://bmvc2019.org/wp-content/uploads/papers/0628-paper.pdf), [**Harmonic Networks with Limited Training Samples**](https://ieeexplore.ieee.org/abstract/document/8902831), [**Harmonic Convolutional Networks based on Discrete Cosine Transform**](https://doi.org/10.1016/j.patcog.2022.108707).

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

## Requirements

Tested on Python 2.7, 3.5, PyTorch 4.1, 1.0, 1.1.
Install the required packages by running:

```
pip install -r requirements.txt
```

## Running the code

To run the code, --dataset and WRN parameters need to be specified such as --depth N, --width K, optional dropout --dropout 0.X or nesterov momentum --nesterov. By default, fully harmonic WRN is trained, which can be changed by importing one of the other network definitions from models folder. Parameter &lambda; from the papers can be set to 2 or 3 specifying --level. Model with progressive compression has its own definition.

For training harmonic WRN-28-10 with lambda compression of level 3 and dropout of 0.3 with 2 GPUs on CIFAR100 dataset run:

```
python main.py --save ./save_location --depth 28 --width 10 --dropout 0.3 --level 3 --dataset CIFAR100 --epochs 200 --epoch_step [60,120,160] --ngpu 2 --gpu_id 0,1
```

To train a network on one of the predefined limited subsets --subset_size (one of 100, 500, 1000) and --subset_id (1-5) have to be specified. Number of epochs and scheduler steps have to be adjusted accordingly. An example of using subset number 2 with 500 samples:

```
python main.py --save ./save_location --depth 16 --width 8 --dropout 0.2 --epochs 4000 --epoch_step [1200,2400,3200] --subset_size 500 --subset_id 2 --ngpu 2 --gpu_id 0,1
```

## Cite

```
@article{Ulicny22,
  title = {Harmonic convolutional networks based on discrete cosine transform},
  journal = {Pattern Recognition},
  volume = {129},
  pages = {108707},
  year = {2022},
  issn = {0031-3203},
  doi = {https://doi.org/10.1016/j.patcog.2022.108707},
  url = {https://www.sciencedirect.com/science/article/pii/S0031320322001881},
  author = {Matej Ulicny and Vladimir A. Krylov and Rozenn Dahyot},
}
```
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

## License

This project is licensed under the BSD License - see the [LICENSE](LICENSE) file for details.
