[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/harmonic-networks-with-limited-training/image-classification-on-stl-10)](https://paperswithcode.com/sota/image-classification-on-stl-10?p=harmonic-networks-with-limited-training)
# Harmonic Wide Residual Networks on STL-10

The code used for experiments in paper [**Harmonic Networks with Limited Training Samples**](https://ieeexplore.ieee.org/abstract/document/8902831).

The implementation is based on the original PyTorch WRN code from https://github.com/szagoruyko/wide-residual-networks/tree/master/pytorch


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

For example to train harmonic WRN of depth 16 and width multiplier 8 with dropout on STL dataset using 2 GPUs run:

```
python main.py --save ./save_location --depth 16 --width 8 --dropout 0.3 --batch_size 32 --ngpu 2 --gpu_id 0,1
```

By default the whole set of 5000 images is used for training. Training on pre-defined fold is possible by specifying --fold parameter (0-9). Number of epochs and scheduler steps are adjusted automatically. Example of training on fold number 4:

```
python main.py --save ./save_location --depth 16 --width 8 --dropout 0.3  --batch_size 32 --fold 4 --ngpu 2 --gpu_id 0,1
```

## Cite

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
