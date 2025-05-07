import numpy as np
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import jax.numpy as jnp
import jax
from models.backbone import TransporterNets
import flax
from models.text_encoder import Encoder
""" 
    TransporterNet evaluate phase
"""

class TransporterNetsEval:
    def __init__(self, checkpoint_path):
        self.TransporterNets = TransporterNets()
        # Coordinate map (i.e. position encoding).
        self.coord_x, self.coord_y = np.meshgrid(np.linspace(-1, 1, 224),
                                                 np.linspace(-1, 1, 224), sparse=False, indexing='ij')
        self.coords = np.concatenate((self.coord_x[..., None], self.coord_y[..., None]), axis=2)
        self.encoder = Encoder(cpu=True)
        self.checkpoint_path = checkpoint_path

    def eval_step(self, params, batch):
        pick_logits, place_logits = self.TransporterNets.apply({'params': params}, batch['img'], batch['text'])
        return pick_logits, place_logits

    def n_params(self, params):
        return jnp.sum(jnp.int32([self.n_params(v) if isinstance(v, dict) or isinstance(v, flax.core.frozen_dict.FrozenDict)
                                  else np.prod(v.shape) for v in params.values()]))

    def load_model(self, load_pretrained=True):
        rng = jax.random.PRNGKey(0)
        rng, key = jax.random.split(rng)
        init_img = jnp.ones((4, 224, 224, 5), jnp.float32)
        init_text = jnp.ones((4, 512), jnp.float32)
        init_pix = jnp.zeros((4, 2), np.int32)
        init_params = self.TransporterNets.init(key, init_img, init_text, init_pix)['params']
        # print(f'Model parameters: {self.n_params(init_params):,}')
        optimizer = optax.adam(learning_rate=1e-4)
        optim = TrainState.create(apply_fn=self.TransporterNets.apply,
                                  params=init_params,
                                  tx=optimizer)
        if load_pretrained:
            ckpt_path = self.checkpoint_path
            self.optim = checkpoints.restore_checkpoint(ckpt_path, optim)
            print('...loading TransporterNet...')

    def eval(self, observation, instruction):
        self.load_model(load_pretrained=True)
        text_feats = self.encoder.encode(instruction)

        # Normalize image and add batch dimension.
        img = observation['image'][None, ...] / 255
        # img = prev_obs[None, ...] / 255
        img = np.concatenate((img, self.coords[None, ...]), axis=3)

        # Run Transporter Nets to get pick and place heatmaps.
        batch = {'img': jnp.float32(img), 'text': jnp.float32(text_feats)}
        pick_map, place_map = self.eval_step(self.optim.params, batch)
        pick_map, place_map = np.float32(pick_map), np.float32(place_map)

        # Get pick position.
        pick_max = np.argmax(np.float32(
            pick_map)).squeeze()  # Ensures that the output is a scalar (removes unnecessary dimensions if needed)
        pick_yx = (pick_max // 224, pick_max % 224)
        pick_yx = np.clip(pick_yx, 20, 204)  # ensures that the coordinates stay within a valid range
        pick_xyz = observation['xyzmap'][pick_yx[0], pick_yx[1]]

        # Get place position.
        place_max = np.argmax(np.float32(place_map)).squeeze()
        place_yx = (place_max // 224, place_max % 224)
        place_yx = np.clip(place_yx, 20, 204)
        place_xyz = observation['xyzmap'][place_yx[0], place_yx[1]]

        # coordinate
        act = {'pick': pick_xyz, 'place': place_xyz}
        return act