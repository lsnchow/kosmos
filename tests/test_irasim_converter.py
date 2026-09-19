from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest

_PATH=Path(__file__).parents[1]/"cluster"/"convert_irasim_checkpoint.py"
_SPEC=importlib.util.spec_from_file_location("irasim_converter",_PATH)
converter=importlib.util.module_from_spec(_SPEC); _SPEC.loader.exec_module(converter)

class ConverterSecurityTests(unittest.TestCase):
    def test_exact_reviewed_globals_are_accepted_without_torch_import(self):
        self.assertEqual(converter.validate_unsafe_globals(converter.APPROVED_GLOBALS),sorted(converter.APPROVED_GLOBALS))
    def test_unknown_pickle_global_is_rejected_before_load(self):
        with self.assertRaises(RuntimeError): converter.validate_unsafe_globals({"evil.module.Payload"})
    def test_omegaconf_aliases_are_inert(self):
        value=converter.InertMetadata(); value.__setstate__({"x":1})
        self.assertEqual(value.state,{"x":1})
        self.assertIn("omegaconf.dictconfig.DictConfig",converter.OMEGACONF_ALIASES)
