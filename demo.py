import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19
import gdown
from PIL import Image
from torchvision import transforms
import gradio as gr

crop_size = 512
model_path = "pretrained_models/model_qnrf.pth"
data_path = "images/"

if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1nnIHPaV9RGqK8JHL645zmRvkNrahD9ru"
    gdown.download(url, model_path, quiet=False)

device = torch.device('cpu')

model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()


def predict(inp):
    inp = Image.fromarray(inp.astype('uint8'), 'RGB')
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    inp = inp.to(device)
    with torch.set_grad_enabled(False):
        outputs, _ = model(inp)
    count = torch.sum(outputs).item()
    return int(count)


title = "Distribution Matching for Crowd Counting"
desc = "A demo of DM-Count, a NeurIPS 2020 paper by Wang et al. Outperforms the state-of-the-art methods by a " \
       "large margin on four challenging crowd counting datasets: UCF-QNRF, NWPU, ShanghaiTech, and UCF-CC50. " \
       "This demo uses the QNRF trained model. Try it by uploading an image or clicking on an example " \
       "(could take up to 30s since its running on CPU)."
examples = [
    "images/1.png",
    "images/2.png",
    "images/3.png",
]
inputs = gr.inputs.Image(label="Image of Crowd")
outputs = gr.outputs.Label(label="Predicted Count")
gr.Interface(fn=predict, inputs=inputs, outputs=outputs, title=title, description=desc, examples=examples, allow_flagging=False).launch()
