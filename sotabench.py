import torch
from torchbench.image_classification import ImageNet
import sys
sys.path.insert(0,'./imagenet/resnext')
from imagenet.resnext.timm import create_model
from imagenet.resnext.timm.data import resolve_data_config, transforms_imagenet_eval, transforms_imagenet_eval
from imagenet.resnext.timm.models import TestTimePoolHead


model = create_model(
    'harm_se_resnext101_64x4d',
    num_classes=1000,
    in_chans=3,
    pretrained=True
)

data_config = resolve_data_config(dict(), model=model, verbose=True)
#if m['ttp']:
#    model = TestTimePoolHead(model, model.default_cfg['pool_size'])
#    data_config['crop_pct'] = 1.0

data_config.update(img_size = data_config['input_size'][2])
del data_config['input_size']
input_transform = transforms_imagenet_eval(**data_config)

# Run the benchmark
ImageNet.benchmark(
    model=model,
    paper_model_name='Harm-SE-RNX(64x4d)',
    paper_arxiv_id='2001.06570',
    input_transform=input_transform,
    batch_size=256,
    num_gpu=1,
    data_root=('~/data/imagenet')
)

torch.cuda.empty_cache()



