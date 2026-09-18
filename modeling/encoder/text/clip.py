import torch
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"


class ClipTokenizer:
    def __init__(self):
        self.tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL_NAME)

    @torch.inference_mode()
    def __call__(self, instructions):
        return self.tokenizer(
            instructions, padding="longest", return_tensors="pt"
        )["input_ids"]


class ClipTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(
            CLIP_MODEL_NAME, use_safetensors=True
        )

    def forward(self, tokens):
        return self.model(tokens).last_hidden_state
