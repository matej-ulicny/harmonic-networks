import torch
from torchbench.image_classification import ImageNet
import sys
sys.path.insert(0,'./imagenet/resnext')
from imagenet.resnext.timm import create_model
from imagenet.resnext.timm.data import resolve_data_config, transforms_imagenet_eval, transforms_imagenet_eval
from imagenet.resnext.timm.models import TestTimePoolHead
# imports for resnets
sys.path.insert(0,'./imagenet')
from imagenet import models
import torchvision.transforms as transforms

model_names = ['resnet50',
               'resnet101',
               'harm_se_resnext101_32x4d', 
               'harm_se_resnext101_32x4d', 
               'harm_se_resnext101_64x4d',
               'harm_se_resnext101_64x4d']
paper_names = ['Harm-ResNet-50', 
               'Harm-ResNet-101',
               'Harm-SE-RNX-101 32x4d', 
               'Harm-SE-RNX-101 32x4d (320x320, Mean-Max Pooling)', 
               'Harm-SE-RNX-101 64x4d', 
               'Harm-SE-RNX-101 64x4d (320x320, Mean-Max Pooling)']
input_sizes = [224, 224, 224, 320, 224, 320]
paper_results = [{'Top 1 Accuracy': 0.7698, 'Top 5 Accuracy': 0.9337},
                 {'Top 1 Accuracy': 0.7852, 'Top 5 Accuracy': 0.9425},
                 {'Top 1 Accuracy': 0.8045, 'Top 5 Accuracy': 0.9521},
                 {'Top 1 Accuracy': 0.8128, 'Top 5 Accuracy': 0.9577},
                 {'Top 1 Accuracy': 0.8164, 'Top 5 Accuracy': 0.09563},
                 {'Top 1 Accuracy': 0.8266, 'Top 5 Accuracy': 0.9629}]

for model_name, paper_name, input_size, paper_result in zip(model_names, paper_names, input_sizes, paper_results):

    if 'resnet' in model_name:

        model = models.__dict__[model_name](pretrained=True, harm_root=True, harm_res_blocks=True)

        input_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        model = create_model(
            model_name,
            num_classes=1000,
            in_chans=3,
            pretrained=True
        )

        data_config = resolve_data_config({'img_size': input_size}, model=model, verbose=True)
        data_config.update(img_size = data_config['input_size'][2])
        del data_config['input_size']
        if input_size > 224:
            model = TestTimePoolHead(model, model.default_cfg['pool_size'])
            data_config['crop_pct'] = 1.0
        input_transform = transforms_imagenet_eval(**data_config)
        
    # Run the benchmark
    ImageNet.benchmark(
        model=model,
        paper_model_name=paper_name,
        paper_arxiv_id='2001.06570',
        paper_pwc_id='harmonic-convolutional-networks-based-on',
        paper_results=paper_result,
        input_transform=input_transform,
        batch_size=256,
        num_gpu=1,
        data_root=('~/data/imagenet')
    )

torch.cuda.empty_cache()

