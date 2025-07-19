import numpy as np
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import jax.numpy as jnp
import jax
from models.backbone import TransporterNets
import flax
from PIL import Image, ImageDraw
import os
from datetime import datetime
import clip
import torch
from models.text_encoder import Encoder

class TransporterNetsEval:
    def __init__(self, checkpoint_path):
        self.TransporterNets = TransporterNets()
        self.coord_x, self.coord_y = np.meshgrid(np.linspace(-1, 1, 224),
                                                 np.linspace(-1, 1, 224), indexing='ij')
        self.coords = np.stack((self.coord_x, self.coord_y), axis=-1)
        self.encoder = Encoder(cpu=True)
        self.checkpoint_path = checkpoint_path
        self.output_dir = "./output_images/"
        os.makedirs(self.output_dir, exist_ok=True)

    def eval_step(self, params, batch):
        pick_logits, place_logits = self.TransporterNets.apply({'params': params}, batch['img'], batch['text'])
        return pick_logits, place_logits

    def n_params(self, params):
        return jnp.sum(
            jnp.int32([self.n_params(v) if isinstance(v, dict) or isinstance(v, flax.core.frozen_dict.FrozenDict)
                       else np.prod(v.shape) for v in params.values()]))

    def load_model(self, load_pretrained=True):
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
        init_text = jnp.ones((4, 512), jnp.float32)
        init_pix = jnp.zeros((4, 2), np.int32)
        init_params = self.TransporterNets.init(key, init_img, init_text, init_pix)['params']
        optimizer = optax.adam(learning_rate=1e-4)
        optim = TrainState.create(apply_fn=self.TransporterNets.apply,
                                  params=init_params,
                                  tx=optimizer)
        if load_pretrained:
            ckpt_path = self.checkpoint_path
            print(ckpt_path)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
            self.optim = checkpoints.restore_checkpoint(ckpt_path, optim)
            print('Loaded checkpoint:', ckpt_path)

    def eval(self, observation, instruction):
        self.load_model(load_pretrained=True)
        text_feats = self.encoder.encode(instruction)
        # print("Text features shape:", text_feats.shape, "min:", text_feats.min(), "max:", text_feats.max())

        img = observation['image'][None, ...] / 255.0
        img = np.concatenate((img, self.coords[None, ...]), axis=3)
        img = jnp.float32(img)
        # print("Input image shape:", img.shape, "dtype:", img.dtype)

        batch = {'img': img, 'text': jnp.float32(text_feats)}
        pick_map, place_map = self.eval_step(self.optim.params, batch)
        pick_map, place_map = np.float32(pick_map), np.float32(place_map)

        pick_max = np.argmax(pick_map.flatten())
        pick_yx = (pick_max // 224, pick_max % 224)
        pick_yx = np.clip(pick_yx, 20, 204)
        place_max = np.argmax(place_map.flatten())
        place_yx = (place_max // 224, place_max % 224)
        place_yx = np.clip(place_yx, 20, 204)

        if 'xyzmap' not in observation:
            print("Warning: xyzmap not provided")
            observation['xyzmap'] = np.zeros((224, 224, 3), dtype=np.float32)
        pick_xyz = observation['xyzmap'][pick_yx[0], pick_yx[1]]
        place_xyz = observation['xyzmap'][place_yx[0], place_yx[1]]
        act = {'pick': pick_xyz, 'place': place_xyz}
        return act