# Object detection with harmonic networks built on mmdetection

Implements object detection models described in [**Harmonic Convolutional Networks based on Discrete Cosine Transform**](https://arxiv.org/abs/2001.06570).

Detection models based on faster R-CNN framework and RetinaNet after 24 epochs (20 for Cascade models). All backbones are transformed into Feature Pyramid Networks (FPNs).

Bounding box performance:

| Model |  Baseline mAP&uarr; | Harmonic mAP&uarr; |
| ----- | :-----------------: | :----------------: |
| RetinaNet ResNet-50 | 36.4 | 36.8 |
| Faster R-CNN ResNet-50 | 37.7 | 38.4 |
| Mask R-CNN ResNet-50 | 38.5 | 38.9 |
| RetinaNet ResNet-101 | 38.1 | 39.2 |
| Faster R-CNN ResNet-101 | 39.3 | 40.3 |
| Mask R-CNN ResNet-101 | 40.3 | 41.5 |
| Cascade Faster R-CNN ResNet-101 | 42.5 | 43.5 |
| Cascade Mask R-CNN ResNet-101 | 43.3 | 44.3 |
| Hybrid Task R-CNN ResNet-101 | 44.9 | 46.0 |

Mask prediction performance:

| Model |  Baseline mAP&uarr; | Harmonic mAP&uarr; |
| ----- | :-----------------: | :----------------: |
| Mask R-CNN ResNet-50 | 35.1 | 35.5 |
| Mask R-CNN ResNet-101 | 36.5 | 37.3 |
| Cascade Mask R-CNN ResNet-101 | 37.6 | 38.3 |
| Hybrid Task R-CNN ResNet-101 | 39.4 | 40.2 |

Repository is cloned from [mmdetection-0.6.0](https://github.com/open-mmlab/mmdetection).

Changelog
* Harmonic ResNet definition extends backbone list
* Config files to reporduce all models in the paper

## Installation

Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Running the code

Models can be trained 
```
./tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [--validate]
```
Model configs have learning rates adjusted for 2 GPU cards, except for the htc config that expects 4 GPUs. Configs for all above-mentioned models are available. For example  training Mask-RCNN model based on our harmonic ResNet-101 on 2 GPUs can be realized as:
``` 
./tools/dist_train.sh configs/mask_rcnn_hr101_fpn_1x.py 2 --validate
```

To train with standard 2x schedule siply double the number of epochs and values of optimizers steps in config files.

## Citation
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
and mmdetection [technical report](https://arxiv.org/abs/1906.07155):
```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```
