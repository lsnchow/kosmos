"""Fused world-model batch-packing contract tests.

These run on CPU against an injected fake pipeline and never import torch,
diffusers, or any model runtime.  They test serialization, grouping, ordering,
seed routing, failure isolation and timing *labels*.  They do not and cannot
establish that the pinned ``Cosmos3OmniPipeline`` accepts the batched
``CosmosActionCondition`` shape, nor that fusing B requests actually divides
GPU-seconds by B: both of those are GPU measurements, not unit tests.
"""

from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from plumb.adapters.contracts import FUSED_BATCH_EQUAL_SHARE, ServerTiming, WorldRequest, WorldResult
from plumb.adapters.worlds import (
    ActionLengthCertification,
    BackendContractError,
    CertificationError,
    COSMOS3_EDGE_MODEL_ID,
    COSMOS3_NANO_MODEL_ID,
    COSMOS_FUSED_BATCH_MODE,
    Cosmos3NanoDiffusersAdapter,
    Cosmos3NanoDiffusersProfile,
    MixedBatchError,
    ProbedActionLength,
    UNBATCHED_SINGLE_FORWARD,
    _CosmosRuntime,
)


# --------------------------------------------------------------------------- fakes


class _FakeTensor(tuple):
    """Nested-tuple stand-in for a stacked tensor; keeps the exact rows."""

    dtype: Optional[str] = None

    @classmethod
    def build(cls, rows: Any, dtype: Any) -> "_FakeTensor":
        tensor = cls(rows)
        tensor.dtype = dtype
        return tensor


class _FakeGenerator:
    """Records only the seed it was given, which is all the fakes need."""

    def __init__(self) -> None:
        self.seed: Optional[int] = None

    def manual_seed(self, seed: int) -> "_FakeGenerator":
        self.seed = int(seed)
        return self


class _FakeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"
    cuda = None

    @staticmethod
    def as_tensor(value: Any, dtype: Any = None) -> _FakeTensor:
        return _FakeTensor.build(value, dtype)

    @staticmethod
    def Generator(device: Any = None) -> _FakeGenerator:  # noqa: N802 - mirrors torch
        return _FakeGenerator()


class _FakeActionCondition:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _frame(prompt: str, seed: int, rows: Sequence[Sequence[float]], index: int) -> bytes:
    """A frame whose bytes depend only on this sample's own inputs.

    Deliberately independent of batch position and batch size: if the adapter
    routed another member's seed, prompt, or action rows into this slot, these
    bytes change and the byte-identity test fails.
    """

    material = "%s|%d|%d|%s" % (prompt, int(seed), int(index), repr(tuple(tuple(row) for row in rows)))
    return hashlib.sha256(material.encode("utf-8")).digest()


class _SeedDrivenPipeline:
    """Fake pipeline handling both the single and the fused call shapes.

    Single shape: ``prompt`` is a string, ``generator`` one generator, and
    ``action.raw_actions`` is ``(N, 10)``.  Fused shape: ``prompt`` is a list of
    B strings, ``generator`` a list of B generators, and ``action.raw_actions``
    is ``(B, N, 10)``.  Each sample's frames come only from that sample's own
    prompt, seed and action rows.
    """

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def _samples(self, kwargs: Dict[str, Any]) -> Tuple[List[Tuple[str, Any, Any]], bool]:
        prompts = kwargs["prompt"]
        generators = kwargs["generator"]
        rows = kwargs["action"].raw_actions
        if isinstance(prompts, str):
            return [(prompts, generators, rows)], False
        if len(prompts) != len(generators) or len(prompts) != len(rows):
            raise AssertionError(
                "fused call must supply one prompt, one generator and one action block per member; got %d/%d/%d"
                % (len(prompts), len(generators), len(rows))
            )
        return list(zip(prompts, generators, rows)), True

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        samples, batched = self._samples(kwargs)
        videos = [
            [_frame(prompt, generator.seed, rows, index) for index in range(len(rows) + 1)]
            for prompt, generator, rows in samples
        ]
        return SimpleNamespace(video=videos if batched else videos[0])


class _WrongCountPipeline(_SeedDrivenPipeline):
    """Returns one sequence too few, to prove frames are never reused."""

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        result = super().__call__(**kwargs)
        return SimpleNamespace(video=result.video[:-1])


class _ExplodingPipeline(_SeedDrivenPipeline):
    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        raise RuntimeError("simulated CUDA out of memory for the whole fused batch")


# --------------------------------------------------------------------------- helpers


def _profile(**overrides: Any) -> Cosmos3NanoDiffusersProfile:
    settings: Dict[str, Any] = {
        "profile_id": "fixture-batch",
        "local_model_path": "/not-needed-for-injected-fixture",
        "allowed_action_lengths": (16,),
        "probe_action_lengths": (1,),
    }
    settings.update(overrides)
    return Cosmos3NanoDiffusersProfile(**settings)


def _adapter(
    profile: Optional[Cosmos3NanoDiffusersProfile] = None,
    pipeline: Optional[_SeedDrivenPipeline] = None,
) -> Tuple[Cosmos3NanoDiffusersAdapter, _SeedDrivenPipeline]:
    resolved_profile = profile if profile is not None else _profile()
    pipe = pipeline if pipeline is not None else _SeedDrivenPipeline()
    runtime = _CosmosRuntime(_FakeTorch, None, _FakeActionCondition)
    adapter = Cosmos3NanoDiffusersAdapter(
        resolved_profile,
        runtime_factory=lambda: runtime,
        pipeline_factory=lambda _profile, _runtime: pipe,
    )
    return adapter, pipe


def _request(
    *,
    count: int = 16,
    seed: int = 7,
    prompt: str = "Close the drawer",
    profile_id: str = "fixture-batch",
    domain: str = "bridge_orig_lerobot",
    width: int = 10,
    offset: float = 0.0,
    request_id: Optional[str] = None,
    timestamps: Optional[Tuple[float, ...]] = None,
) -> WorldRequest:
    rows = tuple(
        tuple(float(column) + offset + row for column in range(width)) for row in range(count)
    )
    return WorldRequest(
        conditioning_image=object(),
        prompt=prompt,
        domain=domain,
        compiled_actions=rows,
        nominal_control_timestamps=(
            timestamps if timestamps is not None else tuple((index + 1) / 5.0 for index in range(count))
        ),
        seed=seed,
        compatibility_profile_id=profile_id,
        request_id=request_id,
    )


def _episode_batch(size: int = 8, count: int = 16) -> Tuple[WorldRequest, ...]:
    """Requests that differ only in per-item data: seed, prompt, actions, id."""

    return tuple(
        _request(
            count=count,
            seed=1000 + index,
            prompt="Task %d: close the drawer" % index,
            offset=float(index),
            request_id="episode-%02d" % index,
        )
        for index in range(size)
    )


# --------------------------------------------------------------------------- tests


class FusedBatchForwardTests(unittest.TestCase):
    def test_batch_of_eight_issues_exactly_one_pipeline_call(self) -> None:
        adapter, pipe = _adapter()
        requests = _episode_batch(8)

        results = adapter.generate_batch(requests)

        self.assertEqual(len(pipe.calls), 1, msg="a fused batch must not loop generate()")
        self.assertEqual(adapter.backend_call_attempts, 1)
        self.assertEqual(adapter.successful_backend_calls, 1)
        self.assertEqual(adapter.fused_batch_calls, 1)
        self.assertEqual(adapter.fused_batched_items, 8)
        self.assertEqual(len(results), 8)
        self.assertTrue(all(isinstance(result, WorldResult) for result in results))

    def test_the_single_call_carries_a_stacked_condition_and_per_sample_seeds(self) -> None:
        adapter, pipe = _adapter()
        requests = _episode_batch(8)

        adapter.generate_batch(requests)
        call = pipe.calls[0]
        condition = call["action"]

        # (B, N, 10): batch leads, then the action rows, then the 10-D Bridge row.
        self.assertEqual(len(condition.raw_actions), 8)
        self.assertEqual(len(condition.raw_actions[0]), 16)
        self.assertEqual(len(condition.raw_actions[0][0]), 10)
        self.assertEqual(condition.raw_actions.dtype, _FakeTorch.float32)
        self.assertEqual(condition.chunk_size, 16)
        self.assertEqual(condition.mode, "forward_dynamics")
        self.assertEqual(condition.domain_name, "bridge_orig_lerobot")
        self.assertEqual(len(condition.image), 8)
        self.assertEqual(
            [id(image) for image in condition.image],
            [id(request.conditioning_image) for request in requests],
        )
        # Action mode passes no top-level image/num_frames, exactly as unbatched.
        self.assertNotIn("image", call)
        self.assertNotIn("num_frames", call)
        # One prompt and one generator per member; seeds stay per request.
        self.assertEqual(call["prompt"], [request.prompt for request in requests])
        self.assertEqual(
            [generator.seed for generator in call["generator"]],
            [request.seed for request in requests],
        )
        self.assertEqual(len(set(id(generator) for generator in call["generator"])), 8)

    def test_results_come_back_in_input_order(self) -> None:
        adapter, _ = _adapter()
        requests = _episode_batch(8)

        results = adapter.generate_batch(requests)

        self.assertEqual(
            [result.request_id for result in results],
            [request.request_id for request in requests],
        )
        self.assertEqual([result.metadata["batch_index"] for result in results], list(range(8)))
        self.assertEqual(
            [result.metadata["per_sample_generator_seed"] for result in results],
            [request.seed for request in requests],
        )

    def test_every_item_gets_n_plus_one_frames_with_the_conditioning_frame_first(self) -> None:
        adapter, _ = _adapter()
        requests = _episode_batch(8)

        results = adapter.generate_batch(requests)

        for request, result in zip(requests, results):
            self.assertTrue(result.conditioning_frame_included)
            self.assertEqual(len(result.frames), len(request.compiled_actions) + 1)
            self.assertEqual(len(result.future_frames), len(request.compiled_actions))
            self.assertEqual(
                result.nominal_frame_timestamps,
                tuple(index / 5.0 for index in range(len(result.frames))),
            )
            self.assertTrue(result.metadata["nominal_timestamps_only"])
            # The same frame contract the unbatched path enforces.
            result.validate(len(request.compiled_actions))

    def test_a_probe_length_batch_still_reports_one_future_frame_per_action(self) -> None:
        adapter, pipe = _adapter()
        requests = tuple(_request(count=1, seed=40 + index) for index in range(4))

        results = adapter.generate_batch(requests)

        self.assertEqual(len(pipe.calls), 1)
        for result in results:
            self.assertEqual(len(result.frames), 2)
            self.assertEqual(len(result.future_frames), 1)
            self.assertEqual(result.metadata["action_length_status"], "probe_unqualified")


class BatchedOutputIdentityTests(unittest.TestCase):
    """Batching must be a scheduling change only, never a numerical one."""

    def test_batched_frames_are_byte_identical_to_solo_generation(self) -> None:
        requests = _episode_batch(8)

        solo_frames = []
        for request in requests:
            solo_adapter, solo_pipe = _adapter()
            solo = solo_adapter.generate(request)
            self.assertEqual(len(solo_pipe.calls), 1)
            self.assertIsInstance(solo_pipe.calls[0]["prompt"], str)
            solo_frames.append(solo.frames)

        batched_adapter, batched_pipe = _adapter()
        batched = batched_adapter.generate_batch(requests)
        self.assertEqual(len(batched_pipe.calls), 1)

        for index, (expected, result) in enumerate(zip(solo_frames, batched)):
            self.assertEqual(
                result.frames,
                expected,
                msg="batch member %d differs from the same request generated alone" % index,
            )
            self.assertEqual(
                [hashlib.sha256(frame).hexdigest() for frame in result.frames],
                [hashlib.sha256(frame).hexdigest() for frame in expected],
            )

    def test_frame_zero_is_handled_identically_to_generate(self) -> None:
        requests = _episode_batch(4)
        solo_adapter, _ = _adapter()
        solo = solo_adapter.generate(requests[2])

        batched_adapter, _ = _adapter()
        batched = batched_adapter.generate_batch(requests)[2]

        self.assertEqual(batched.frames[0], solo.frames[0])
        self.assertEqual(batched.conditioning_frame_included, solo.conditioning_frame_included)
        self.assertEqual(batched.nominal_frame_timestamps, solo.nominal_frame_timestamps)
        self.assertEqual(batched.future_frames, solo.future_frames)

    def test_a_members_frames_depend_on_its_own_seed_not_its_neighbours(self) -> None:
        base = _episode_batch(4)
        adapter, _ = _adapter()
        original = adapter.generate_batch(base)

        # Change only member 0's seed; nobody else's frames may move.
        changed = (_request(count=16, seed=999, prompt=base[0].prompt, offset=0.0),) + base[1:]
        other_adapter, _ = _adapter()
        after = other_adapter.generate_batch(changed)

        self.assertNotEqual(after[0].frames, original[0].frames)
        for index in range(1, 4):
            self.assertEqual(after[index].frames, original[index].frames)

    def test_generate_still_reports_an_unbatched_forward(self) -> None:
        adapter, pipe = _adapter()
        result = adapter.generate(_request())

        self.assertEqual(result.metadata["batch_execution_mode"], UNBATCHED_SINGLE_FORWARD)
        self.assertEqual(result.metadata["batch_size"], 1)
        self.assertIsNone(result.timing.batch_size)
        self.assertIsNone(result.timing.attributed_gpu_seconds)
        self.assertFalse(result.timing.is_shared_backend_call)
        self.assertEqual(len(pipe.calls[0]["action"].raw_actions[0]), 10)
        self.assertEqual(adapter.fused_batch_calls, 0)


class BatchGroupingTests(unittest.TestCase):
    def test_batch_key_ignores_per_item_data_and_splits_on_shape_and_schedule(self) -> None:
        profile = _profile()
        base = _request()

        same_key_variants = (
            _request(seed=99),
            _request(prompt="Put the eggplant in the sink"),
            _request(request_id="another-episode"),
        )
        for variant in same_key_variants:
            self.assertEqual(
                Cosmos3NanoDiffusersAdapter.batch_key(variant, profile),
                Cosmos3NanoDiffusersAdapter.batch_key(base, profile),
                msg="seeds, prompts and ids are per-item data and must still fuse",
            )

        self.assertNotEqual(
            Cosmos3NanoDiffusersAdapter.batch_key(_request(count=4), profile),
            Cosmos3NanoDiffusersAdapter.batch_key(base, profile),
        )
        self.assertNotEqual(
            Cosmos3NanoDiffusersAdapter.batch_key(_request(domain="other_domain"), profile),
            Cosmos3NanoDiffusersAdapter.batch_key(base, profile),
        )
        self.assertNotEqual(
            Cosmos3NanoDiffusersAdapter.batch_key(_request(profile_id="other-profile"), profile),
            Cosmos3NanoDiffusersAdapter.batch_key(base, profile),
        )
        for field, value in (
            ("resolution_tier", 480),
            ("num_inference_steps", 12),
            ("guidance_scale", 3.0),
            ("fps", 10.0),
            ("view_point", "third_person_view"),
        ):
            self.assertNotEqual(
                Cosmos3NanoDiffusersAdapter.batch_key(base, _profile(**{field: value})),
                Cosmos3NanoDiffusersAdapter.batch_key(base, profile),
                msg="%s changes the forward and must split the batch" % field,
            )

    def test_batch_key_without_a_profile_is_stable_but_differently_spelled(self) -> None:
        profile = _profile()
        loose = Cosmos3NanoDiffusersAdapter.batch_key(_request())
        self.assertEqual(loose, Cosmos3NanoDiffusersAdapter.batch_key(_request(seed=3)))
        self.assertIn("profile-implied", loose)
        self.assertNotEqual(loose, Cosmos3NanoDiffusersAdapter.batch_key(_request(), profile))

    def test_a_mixed_action_length_batch_raises_instead_of_splitting(self) -> None:
        adapter, pipe = _adapter(_profile(allowed_action_lengths=(4, 16)))
        requests = (_request(count=16), _request(count=16), _request(count=4))

        with self.assertRaises(MixedBatchError) as caught:
            adapter.generate_batch(requests)

        self.assertIn("batch key", str(caught.exception))
        self.assertEqual(len(pipe.calls), 0, msg="a mixed batch must not be silently split")
        self.assertEqual(adapter.backend_call_attempts, 0)
        self.assertEqual(adapter.fused_batch_calls, 0)

    def test_a_mixed_domain_batch_raises(self) -> None:
        adapter, pipe = _adapter()
        requests = (_request(), _request(domain="other_domain"))

        with self.assertRaises(MixedBatchError):
            adapter.generate_batch(requests)
        self.assertEqual(len(pipe.calls), 0)

    def test_a_mixed_profile_batch_raises(self) -> None:
        adapter, pipe = _adapter()
        requests = (_request(), _request(profile_id="a-different-profile"))

        with self.assertRaises(MixedBatchError):
            adapter.generate_batch(requests)
        self.assertEqual(len(pipe.calls), 0)

    def test_a_batch_of_one_is_legal_and_reports_a_batch_size_of_one(self) -> None:
        adapter, pipe = _adapter()
        request = _request(seed=11)

        results = adapter.generate_batch((request,))

        self.assertEqual(len(results), 1)
        self.assertEqual(len(pipe.calls), 1)
        self.assertEqual(results[0].timing.batch_size, 1)
        self.assertFalse(results[0].timing.is_shared_backend_call)
        self.assertAlmostEqual(
            results[0].timing.attributed_gpu_seconds, results[0].timing.wall_seconds, places=12
        )
        # A batch of one still goes through the batched call shape.
        self.assertEqual(pipe.calls[0]["prompt"], [request.prompt])
        self.assertEqual(len(pipe.calls[0]["action"].raw_actions), 1)

    def test_an_empty_batch_raises_rather_than_reporting_a_call(self) -> None:
        adapter, pipe = _adapter()

        with self.assertRaises(BackendContractError):
            adapter.generate_batch(())
        self.assertEqual(len(pipe.calls), 0)
        self.assertEqual(adapter.backend_call_attempts, 0)


class BatchFailureIsolationTests(unittest.TestCase):
    def test_one_invalid_item_does_not_lose_the_others(self) -> None:
        adapter, pipe = _adapter()
        requests = (
            _request(seed=1, request_id="good-a"),
            # 7-D IRASim rows: invalid for Cosmos, and a per-item data failure
            # rather than a second legitimate group.
            _request(seed=2, width=7, request_id="bad-width"),
            _request(seed=3, request_id="good-b"),
            _request(seed=4, request_id="good-c"),
        )

        results = adapter.generate_batch(requests)

        self.assertEqual(len(results), 4, msg="no input may be dropped")
        self.assertIsInstance(results[1], BackendContractError)
        self.assertIn("10-D", str(results[1]))
        for index in (0, 2, 3):
            self.assertIsInstance(results[index], WorldResult)
            self.assertEqual(len(results[index].frames), 17)
        self.assertEqual(len(pipe.calls), 1)
        # The invalid member never entered the forward, so the fused size is 3.
        self.assertEqual(len(pipe.calls[0]["action"].raw_actions), 3)
        self.assertEqual(results[0].metadata["batch_size"], 3)
        self.assertEqual(results[0].metadata["batch_submitted_size"], 4)
        self.assertEqual(results[0].timing.batch_size, 3)
        self.assertEqual(adapter.fused_batched_items, 3)

    def test_a_malformed_timestamp_item_is_isolated_too(self) -> None:
        adapter, _ = _adapter()
        requests = (
            _request(seed=1),
            _request(seed=2, timestamps=tuple(0.2 for _ in range(16))),
            _request(seed=3),
        )

        results = adapter.generate_batch(requests)

        self.assertIsInstance(results[1], ValueError)
        self.assertIn("strictly increasing", str(results[1]))
        self.assertIsInstance(results[0], WorldResult)
        self.assertIsInstance(results[2], WorldResult)

    def test_a_batch_of_only_invalid_items_makes_no_call_at_all(self) -> None:
        adapter, pipe = _adapter()
        requests = (_request(width=7, seed=1), _request(width=7, seed=2))

        results = adapter.generate_batch(requests)

        self.assertEqual(len(results), 2)
        self.assertTrue(all(isinstance(result, BackendContractError) for result in results))
        self.assertEqual(len(pipe.calls), 0)
        self.assertEqual(adapter.backend_call_attempts, 0)
        self.assertEqual(adapter.fused_batch_calls, 0)

    def test_a_whole_batch_pipeline_failure_raises_and_is_not_counted_as_a_result(self) -> None:
        adapter, _ = _adapter(pipeline=_ExplodingPipeline())

        with self.assertRaises(RuntimeError):
            adapter.generate_batch(_episode_batch(4))

        self.assertEqual(adapter.backend_call_attempts, 1)
        self.assertEqual(adapter.fused_batch_calls, 0)
        self.assertEqual(adapter.successful_backend_calls, 0)

    def test_a_short_result_set_is_refused_rather_than_padded_with_reused_frames(self) -> None:
        adapter, _ = _adapter(pipeline=_WrongCountPipeline())

        with self.assertRaises(BackendContractError) as caught:
            adapter.generate_batch(_episode_batch(4))

        self.assertIn("3 frame sequences for a batch of 4", str(caught.exception))
        self.assertEqual(adapter.successful_backend_calls, 0)


class BatchTimingAttributionTests(unittest.TestCase):
    def test_per_item_gpu_seconds_are_labelled_an_attribution_not_a_measurement(self) -> None:
        adapter, _ = _adapter()
        results = adapter.generate_batch(_episode_batch(8))

        timing = results[0].timing
        self.assertEqual(timing.batch_size, 8)
        self.assertEqual(timing.backend_calls, 1)
        self.assertTrue(timing.is_shared_backend_call)
        self.assertEqual(timing.attribution_method, FUSED_BATCH_EQUAL_SHARE)
        self.assertIsNotNone(timing.wall_seconds)
        self.assertAlmostEqual(timing.attributed_gpu_seconds, timing.wall_seconds / 8.0, places=12)

        payload = timing.as_dict()
        self.assertEqual(payload["attribution_method"], FUSED_BATCH_EQUAL_SHARE)
        self.assertTrue(payload["is_shared_backend_call"])
        self.assertFalse(
            payload["wall_seconds_is_per_item_measurement"],
            msg="the fused wall time is shared and must not read as a per-item measurement",
        )

        metadata = results[0].metadata
        self.assertEqual(metadata["batch_execution_mode"], COSMOS_FUSED_BATCH_MODE)
        self.assertEqual(metadata["attributed_gpu_seconds_method"], FUSED_BATCH_EQUAL_SHARE)
        self.assertFalse(metadata["attributed_gpu_seconds_is_measurement"])
        self.assertTrue(metadata["gpu_peak_memory_is_whole_batch"])
        self.assertFalse(
            metadata["batched_condition_shape_verified"],
            msg="only a real GPU run may flip the batched-shape flag",
        )
        self.assertTrue(any("attribution" in note for note in metadata["batch_limitations"]))

    def test_the_shared_wall_time_is_identical_across_members_and_sums_correctly(self) -> None:
        adapter, _ = _adapter()
        requests = _episode_batch(8)
        results = adapter.generate_batch(requests)

        walls = {result.timing.wall_seconds for result in results}
        self.assertEqual(len(walls), 1, msg="one fused call has exactly one wall time")
        fused_seconds = walls.pop()
        attributed = sum(result.timing.attributed_gpu_seconds for result in results)
        self.assertAlmostEqual(attributed, fused_seconds, places=9)

        expected_key = Cosmos3NanoDiffusersAdapter.batch_key(requests[0], adapter.profile)
        self.assertEqual({result.metadata["batch_key"] for result in results}, {expected_key})

    def test_an_attribution_cannot_be_recorded_without_naming_its_method(self) -> None:
        with self.assertRaises(ValueError):
            ServerTiming(
                backend_calls=1,
                wall_seconds=0.5,
                cold_start=False,
                batch_size=8,
                attributed_gpu_seconds=0.0625,
            )
        with self.assertRaises(ValueError):
            ServerTiming(
                backend_calls=1,
                wall_seconds=0.5,
                cold_start=False,
                attribution_method=FUSED_BATCH_EQUAL_SHARE,
            )
        with self.assertRaises(ValueError):
            ServerTiming(backend_calls=1, wall_seconds=0.5, cold_start=False, batch_size=0)


class SpeedArmLabellingTests(unittest.TestCase):
    def test_the_edge_profile_labels_itself_as_edge_not_nano(self) -> None:
        edge = _profile(
            profile_id="plumb-cosmos3_edge-fd-r256",
            model_id=COSMOS3_EDGE_MODEL_ID,
            model_revision="f" * 40,
        )
        backend_profile = edge.as_backend_profile()

        self.assertEqual(backend_profile.model_id, COSMOS3_EDGE_MODEL_ID)
        self.assertNotEqual(backend_profile.model_id, COSMOS3_NANO_MODEL_ID)

        adapter, _ = _adapter(edge)
        self.assertEqual(adapter.capability().details["model_id"], COSMOS3_EDGE_MODEL_ID)

        solo = adapter.generate(_request(profile_id="plumb-cosmos3_edge-fd-r256"))
        self.assertEqual(solo.metadata["model_id"], COSMOS3_EDGE_MODEL_ID)

        batched_adapter, _ = _adapter(edge)
        batched = batched_adapter.generate_batch(
            tuple(
                _request(profile_id="plumb-cosmos3_edge-fd-r256", seed=index, offset=float(index))
                for index in range(4)
            )
        )
        self.assertEqual({result.metadata["model_id"] for result in batched}, {COSMOS3_EDGE_MODEL_ID})

    def test_the_default_arm_is_still_nano_and_an_empty_label_is_refused(self) -> None:
        self.assertEqual(_profile().model_id, COSMOS3_NANO_MODEL_ID)
        self.assertEqual(_profile().as_backend_profile().model_id, COSMOS3_NANO_MODEL_ID)
        with self.assertRaises(ValueError):
            _profile(model_id="")

    def test_the_model_id_reaches_the_backend_profile_hash_input(self) -> None:
        nano = _profile().as_backend_profile()
        edge = _profile(model_id=COSMOS3_EDGE_MODEL_ID).as_backend_profile()
        self.assertNotEqual(nano.model_id, edge.model_id)


class CertifiedLengthOneTests(unittest.TestCase):
    """Length 1 must be *representable*; whether Cosmos supports it is measured."""

    @staticmethod
    def _certification(*lengths: ProbedActionLength, profile_id: str = "fixture-batch") -> ActionLengthCertification:
        return ActionLengthCertification(
            certification_id="cert-batch-1",
            profile_id=profile_id,
            backend="cosmos3_diffusers",
            domain="bridge_orig_lerobot",
            lengths=tuple(lengths),
            source_uri="fixture://certification.json",
            source_sha256="a" * 64,
        )

    def test_a_certification_including_length_one_flows_into_allowed_lengths(self) -> None:
        certification = self._certification(
            ProbedActionLength(1, "supported", "fixture://n1", "b" * 64, returned_frame_count=2),
            ProbedActionLength(16, "supported", "fixture://n16", "c" * 64, returned_frame_count=17),
        )
        self.assertEqual(certification.supported_lengths, (1, 16))

        # Direct construction keeps working even against the default probe list,
        # because a certified length stops being an open probe.
        profile = _profile(allowed_action_lengths=(1, 16), action_length_certification=certification)
        self.assertEqual(profile.allowed_action_lengths, (1, 16))
        self.assertEqual(profile.probe_action_lengths, ())
        self.assertEqual(profile.probe_lengths_superseded_by_certification, (1,))
        self.assertEqual(profile.action_length_certification_class, "gate_a_certified")
        self.assertEqual(
            profile.as_backend_profile().metadata["probe_lengths_superseded_by_certification"], [1]
        )

        from_certification = Cosmos3NanoDiffusersProfile.from_certification(
            certification, local_model_path="/injected", probe_action_lengths=(1, 4)
        )
        self.assertEqual(from_certification.allowed_action_lengths, (1, 16))
        self.assertEqual(from_certification.probe_action_lengths, (4,))

    def test_a_certified_length_one_profile_fuses_one_action_requests(self) -> None:
        certification = self._certification(
            ProbedActionLength(1, "supported", "fixture://n1", "b" * 64, returned_frame_count=2),
            ProbedActionLength(16, "supported", "fixture://n16", "c" * 64, returned_frame_count=17),
        )
        profile = _profile(allowed_action_lengths=(1, 16), action_length_certification=certification)
        adapter, pipe = _adapter(profile)

        results = adapter.generate_batch(tuple(_request(count=1, seed=50 + index) for index in range(8)))

        self.assertEqual(len(pipe.calls), 1)
        self.assertEqual(len(pipe.calls[0]["action"].raw_actions), 8)
        self.assertEqual(pipe.calls[0]["action"].chunk_size, 1)
        for result in results:
            self.assertEqual(len(result.frames), 2)
            self.assertEqual(result.metadata["action_length_status"], "configured_unqualified")

    def test_a_measured_length_one_failure_stays_unsupported_and_stays_probeable(self) -> None:
        # HANDOFF job 937398 recorded Cosmos N=1 returning only the conditioning
        # frame.  That is a measured failure and is recorded as one.
        certification = self._certification(
            ProbedActionLength(
                1,
                "unsupported",
                "fixture://n1",
                "b" * 64,
                returned_frame_count=1,
                job_id="937398",
                note="returned only the conditioning frame",
            ),
            ProbedActionLength(16, "supported", "fixture://n16", "c" * 64, returned_frame_count=17),
        )
        self.assertEqual(certification.supported_lengths, (16,))
        self.assertEqual(certification.unsupported_lengths, (1,))

        profile = _profile(allowed_action_lengths=(16,), action_length_certification=certification)
        self.assertEqual(profile.allowed_action_lengths, (16,))
        self.assertEqual(profile.probe_action_lengths, (1,))
        self.assertEqual(profile.probe_lengths_superseded_by_certification, ())
        entry = [item for item in certification.as_dict()["lengths"] if item["action_length"] == 1][0]
        self.assertEqual(entry["returned_frame_count"], 1)
        self.assertEqual(entry["expected_frame_count"], 2)
        self.assertEqual(entry["job_id"], "937398")

    def test_a_length_one_probe_that_returned_one_frame_cannot_be_called_supported(self) -> None:
        with self.assertRaises(CertificationError):
            ProbedActionLength(1, "supported", "fixture://n1", "b" * 64, returned_frame_count=1)

    def test_a_certified_profile_exposes_length_one_to_the_rollout_profile_builder(self) -> None:
        certification = self._certification(
            ProbedActionLength(1, "supported", "fixture://n1", "b" * 64, returned_frame_count=2),
            ProbedActionLength(16, "supported", "fixture://n16", "c" * 64, returned_frame_count=17),
        )
        profile = _profile(allowed_action_lengths=(1, 16), action_length_certification=certification)
        adapter, _ = _adapter(profile)

        from plumb.rollout import RolloutController

        rollout_profile = RolloutController()._world_profile(adapter)

        self.assertEqual(rollout_profile.supported_action_lengths, (1, 16))
        self.assertEqual(rollout_profile.certification_class, "gate_a_certified")
        # One-action-per-call cadence (Octo with temporal ensembling) is now
        # representable, and so is every task horizon.
        for horizon in (70, 80, 100):
            self.assertTrue(rollout_profile.represents_exactly(horizon))


if __name__ == "__main__":  # pragma: no cover - mirrors tests/test_adapters.py
    unittest.main()
