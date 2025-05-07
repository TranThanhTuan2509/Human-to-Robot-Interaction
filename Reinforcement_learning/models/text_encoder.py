import clip
import numpy as np
import torch

"""
    Text encoder using pretrained CLIP
"""

class Encoder:
    def __init__(self, cpu=True):
        self.faction = 0.9
        self.cpu = cpu

    def encode(self, text):
        #@markdown Load CLIP model.
        #GPU code
        torch.cuda.set_per_process_memory_fraction(self.faction, None)
        clip_model, clip_preprocess = clip.load("ViT-B/32")
        clip_model.cuda().eval()

        #TPU code
        #import torch_xla
        #import torch_xla.core.xla_model as xm
        #import clip

        # Load the CLIP model and preprocess
        #clip_model, clip_preprocess = clip.load("ViT-B/32")

        # Move the model to TPU
        #device = xm.xla_device()
        #clip_model = clip_model.to(device).eval()

        # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
        # print("Input resolution:", clip_model.visual.input_resolution)
        # print("Context length:", clip_model.context_length)
        # print("Vocab size:", clip_model.vocab_size)
        print('...encoding...')

        text_tokens = clip.tokenize(text).cuda()
        with torch.no_grad():
            text_feats = clip_model.encode_text(text_tokens).float()
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
        if self.cpu:
            text_feats = np.float32(text_feats.cpu()) #text feats shape:  (1, 512)

        return text_feats