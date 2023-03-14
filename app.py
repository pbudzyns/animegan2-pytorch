import pathlib

import gradio as gr
from PIL import Image
import torch

from model import Generator


DEVICE = "cuda"

checkpoints = pathlib.Path("weights")
models = {}

for checkpoint in checkpoints.iterdir():
    net = Generator()
    net.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    net.to(DEVICE).eval()

    models[checkpoint.stem] = net


face2paint = torch.hub.load(
    'AK391/animegan2-pytorch:main',
    'face2paint',
    size=512,
    device="cuda",
    side_by_side=False,
)


def inference(img, ver):
    return face2paint(models[ver], img)


models_options = list(models.keys())
interface = gr.Interface(
    inference,
    [
        gr.inputs.Image(type="pil"),
        gr.inputs.Radio(
            models_options,
            default=models_options[0],
            label='version',
        )
    ],
    gr.outputs.Image(type="pil"),
    title="AnimeGANv2",
    allow_screenshot=False
)

interface.launch(server_name='0.0.0.0')
