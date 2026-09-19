from __future__ import annotations

import unittest

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from plumb.adapters.irasim_runtime import IRASimOneStepAdapter, IRASimOneStepProfile


class _LatentDist:
    def sample(self, generator=None):
        self.generator = generator
        return torch.zeros((1, 4, 32, 40))

class _Vae:
    class config:
        scaling_factor = 0.18215
    def encode(self, value):
        self.encoded_shape = tuple(value.shape)
        return type("Encoded", (), {"latent_dist": _LatentDist()})()

class _Pipeline:
    def __init__(self):
        self.vae = _Vae(); self.calls=[]
    def __call__(self, actions, **kwargs):
        self.calls.append((actions, kwargs))
        return torch.zeros((1, 2, 3, 256, 320)), None

@unittest.skipUnless(torch is not None, "Tensor-layout tests run in the cluster ML environment; no laptop Torch download")
class IRASimRuntimeTests(unittest.TestCase):
    def test_one_step_keeps_16_frame_model_but_calls_video_length_two(self):
        pipe=_Pipeline()
        profile=IRASimOneStepProfile("test","/repo","/ckpt","/vae","/scheduler","/config",device="cpu")
        adapter=IRASimOneStepAdapter(profile, loader=lambda _: (torch, pipe))
        output=adapter.generate_one_step(np.zeros((256,320,3),dtype=np.uint8), [0.1]*7, seed=4)
        actions, kwargs=pipe.calls[0]
        self.assertEqual(profile.num_frames,16)
        self.assertEqual(tuple(actions.shape),(1,1,7))
        self.assertEqual(tuple(kwargs["mask_x"].shape),(1,1,4,32,40))
        self.assertEqual(kwargs["video_length"],2)
        self.assertIsInstance(kwargs["device"], torch.device)
        self.assertEqual(len(output.frames),2)
        self.assertEqual(output.native_action_scaled[:6],(2.0,)*6)
        self.assertEqual(output.native_action_scaled[6],0.1)
        self.assertFalse(output.condition_preprocessing["resize_applied"])

    def test_profile_rejects_architecture_changes(self):
        with self.assertRaises(ValueError):
            IRASimOneStepProfile("x","r","c","v","s","cfg",num_frames=2)

    def test_condition_resize_is_explicit_and_recorded(self):
        pipe=_Pipeline()
        adapter=IRASimOneStepAdapter(IRASimOneStepProfile("test","r","c","v","s","cfg",device="cpu"), loader=lambda _: (torch,pipe))
        out=adapter.generate_one_step(np.zeros((128,160,3),dtype=np.uint8), [0.0]*7)
        self.assertTrue(out.condition_preprocessing["resize_applied"])
        self.assertEqual(out.condition_preprocessing["resized_to_hw"],[256,320])

    def test_condition_posterior_is_reproducible_for_the_request_seed(self):
        class RandomPosterior:
            def sample(self, generator=None):
                return torch.randn((1, 4, 32, 40), generator=generator)

        pipe = _Pipeline()
        pipe.vae.encode = lambda value: type("Encoded", (), {"latent_dist": RandomPosterior()})()
        adapter = IRASimOneStepAdapter(
            IRASimOneStepProfile("test", "r", "c", "v", "s", "cfg", device="cpu"),
            loader=lambda _: (torch, pipe))
        image = np.zeros((256, 320, 3), dtype=np.uint8)
        for seed in (23, 23, 24):
            adapter.generate_one_step(image, [0.0] * 7, seed=seed)
        assert torch.equal(pipe.calls[0][1]["mask_x"], pipe.calls[1][1]["mask_x"])
        assert not torch.equal(pipe.calls[0][1]["mask_x"], pipe.calls[2][1]["mask_x"])
