""" Crowd counting model using Cog """
# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile

# suppress warnings
import warnings

import cv2
import gdown
import numpy as np
import scipy
import torch
from cog import BaseModel, BasePredictor, Input, Path
from PIL import Image
from torchvision import transforms

from models import vgg19

warnings.filterwarnings("ignore")


class Output(BaseModel):
    output: Path
    count: int


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.device = torch.device("cpu")  # device can be "cpu" or "gpu"

        self.models = {}

        # load all models, one for each dataset trained
        for i, name in enumerate(["nwpu", "qnrf", "sh_A", "sh_B"]):
            print(f"Loading model {i}/4: {name}.....")
            self.models[name] = vgg19()
            self.models[name].to(self.device)
            model_path = f"pretrained_models/model_{name}.pth"
            self.models[name].load_state_dict(torch.load(model_path, self.device))
            self.models[name].eval()

    def predict(
        self,
        image: Path = Input(description="Input image"),
        model: str = Input(
            description="Choose which pretrained model to use, each one is trained on a different crowd-counting dataset. Choose between UCF-QNRF (qnrf), NWPU (nwpu), Shanghaitech part A (sh_A) and Shanghaitech part B (sh_B).",
            choices=["qnrf", "nwpu", "sh_A", "sh_B"],
            default="qnrf",
        ),
    ) -> Output:
        """Run a single prediction on the model"""

        image = str(image)
        model = str(model)

        inp = cv2.imread(image)

        inp = Image.fromarray(inp.astype("uint8"), "RGB")
        inp = transforms.ToTensor()(inp).unsqueeze(0)
        inp = inp.to(self.device)

        self.model = self.models[model]
        print(f"Using model {model}......")

        with torch.set_grad_enabled(False):
            outputs, _ = self.model(inp)
        count = torch.sum(outputs).item()
        vis_img = outputs[0, 0].cpu().numpy()
        # normalize density map values from 0 to 1, then map it to 0-255.
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

        # resize output
        h,w,_ = vis_img.shape
        vis_img = cv2.resize(vis_img, (w*3,h*3), interpolation = cv2.INTER_AREA)

        print('Saving output......')
        output_path = Path(tempfile.mkdtemp()) / "predicted_density_map.png"
        cv2.imwrite(str(output_path), vis_img)
        
        print(f"Predicted count: {int(count)}")
        return Output(output=output_path, count=int(count))
