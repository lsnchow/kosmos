from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).parents[1]/"cluster"/"irasim_probe_suite.py";spec=importlib.util.spec_from_file_location("irasim_probe",path);probe=importlib.util.module_from_spec(spec);sys.modules["irasim_probe"]=probe;spec.loader.exec_module(probe)

class IRASimProbeTests(unittest.TestCase):
    def test_case_controls_hold_source_gripper_and_are_bounded(self):
        cases=probe.build_cases((.1,.2,.3,.4,.5,.6,.7),.005)
        self.assertEqual(len(cases),9)
        stationary=next(c for c in cases if c.name=="stationary")
        self.assertEqual(stationary.action,(0.,0.,0.,0.,0.,0.,.7))
        directed=[c for c in cases if c.kind=="directed_axis"]
        self.assertEqual(len(directed),6);self.assertTrue(all(c.action[6]==.7 for c in directed))
        self.assertTrue(all("unqualified" in probe.LABEL for _ in cases))
    def test_bad_source_action_rejected(self):
        with self.assertRaises(ValueError):probe.build_cases((0.,)*6,.01)
    def test_preflight_rejects_malformed_bounds_and_existing_output_before_model(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); out=root/"out"
            with self.assertRaises(ValueError): probe.validate_inputs([0.]*6,{},root/"image",root/"ckpt","0"*64,(0,),.01,root,out)
            with self.assertRaises(ValueError): probe.validate_inputs([0.,0.,0.,0.,0.,0.,2.],{},root/"image",root/"ckpt","0"*64,(0,),.01,root,out)
            out.mkdir()
            with self.assertRaises(FileExistsError): probe.validate_inputs([0.]*7,{},root/"image",root/"ckpt","0"*64,(0,),.01,root,out)
    def test_case_failure_is_recorded_and_later_cases_continue(self):
        class Profile: profile_id="fake"
        class Fake:
            profile=Profile()
            def generate_one_step(self,*args): raise RuntimeError("fixture failure")
        with tempfile.TemporaryDirectory() as d:
            reports=probe.run_cases(Fake(),object(),[probe.Case("one",(0.,)*7,"test","x"),probe.Case("two",(0.,)*7,"test","x")],[0],Path(d),{},)
            self.assertEqual([r["status"] for r in reports],["failed","failed"])
            self.assertTrue((Path(d)/"seed-0-two.json").is_file())
def test_provenance_hash_accepts_bare_or_explicit_sha256():
    from cluster.irasim_probe_suite import normalized_sha
    assert normalized_sha("a" * 64) == normalized_sha("sha256:" + "a" * 64)
    import pytest
    with pytest.raises(ValueError):
        normalized_sha("not-a-digest")


def test_irasim_probe_covers_only_two_canonical_gate_b_arms():
    """Pin which Gate-B control arms this separate IRASim diagnostic supplies.

    The IRASim path is backend-specific and cannot clear Cosmos's Gate B. This
    test records the arms it does *not* cover so a passing IRASim suite can
    never be mistaken for full Gate-B control coverage; the Cosmos arms live in
    ``plumb.adapters.probe_suite``.
    """

    from plumb.gates import REQUIRED_GATE_B_ARMS

    cases = probe.build_cases((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7), 0.005)
    names = {case.name for case in cases}
    covered = {arm for arm in REQUIRED_GATE_B_ARMS if arm in names}
    assert covered == {"original"}
    assert sorted(set(REQUIRED_GATE_B_ARMS) - covered) == [
        "cross_episode",
        "sign_reversed",
        "temporally_permuted",
        "zero",
    ]
    # It does supply the legitimate-stationary control required by spec 0.
    assert "stationary" in names
    assert "unqualified" in probe.LABEL
