import os, sys
import torch
import numpy as np

from models import build_DABDETR, build_dab_deformable_detr
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops

# model_config_path = "model_zoo/DAB_DETR/R50/config.json" # change the path of the model config file
# model_checkpoint_path = "model_zoo/DAB_DETR/R50/checkpoint.pth" # change the path of the model checkpoint
model_config_path = "DABDETR_model_zoo/DAB_Deformable_DETR/R50_v2/config.json" # change the path of the model config file
model_checkpoint_path = "DABDETR_model_zoo/DAB_Deformable_DETR/R50_v2/checkpoint.pth" # change the path of the model checkpoint
# See our Model Zoo section in README.md for more details about our pretrained models.

args = SLConfig.fromfile(model_config_path)
model, criterion, postprocessors = build_dab_deformable_detr(args)
checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])

from PIL import Image
import datasets.transforms as T

image = Image.open("./figure/idea.jpg").convert("RGB")

# transform images
transform = T.Compose([
    T.RandomResize([800], max_size=1333),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image, _ = transform(image, None)

# predict images
output = model(image[None])
output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]))[0]