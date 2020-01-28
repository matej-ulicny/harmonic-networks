import torch
from torchbench.image_classification import ImageNet
import sys
sys.path.insert(0,'./imagenet/resnext')
from imagenet.resnext.timm import create_model
from imagenet.resnext.timm.data import resolve_data_config, transforms_imagenet_eval, transforms_imagenet_eval
from imagenet.resnext.timm.models import TestTimePoolHead


model_names = ['harm_se_resnext101_32x4d', 'harm_se_resnext101_64x4d']
paper_names = ['Harm-SE-RNX(32x4d)', 'Harm-SE-RNX(64x4d)']
input_sizes = [224, 320]

for model_name, paper_name in zip(model_names, paper_names):

    for input_size in input_sizes:

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
            paper_name += ' (320x320, Mean-Max Pooling)'
        input_transform = transforms_imagenet_eval(**data_config)
        
        # Run the benchmark
        ImageNet.benchmark(
            model=model,
            paper_model_name=paper_name,
            paper_arxiv_id='2001.06570',
            input_transform=input_transform,
            batch_size=256,
            num_gpu=1,
            data_root=('~/data/imagenet')
        )
        
torch.cuda.empty_cache()



