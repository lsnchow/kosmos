import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("stream_baseten_chain", Path(__file__).parents[1] / "scripts" / "stream_baseten_chain.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_only_chain_stream_endpoints_are_allowed():
    for path in ("production", "development", "deployment/abc123", "environments/demo-live"):
        url = "https://chain-abc123.api.baseten.co/" + path + "/run_remote"
        assert MODULE.validate_chain_url(url) == url


@pytest.mark.parametrize("url", [
    "http://chain-abc.api.baseten.co/production/run_remote",
    "https://model-abc.api.baseten.co/production/predict",
    "https://chain-abc.api.baseten.co.evil.test/production/run_remote",
    "https://key@chain-abc.api.baseten.co/production/run_remote",
    "https://chain-abc.api.baseten.co/production/run_remote?token=x",
    "https://chain-abc.api.baseten.co:444/production/run_remote",
    "https://chain-abc.api.baseten.co/production/async_run_remote",
])
def test_rejects_unexpected_credentials_destinations_and_nonstream_routes(url):
    with pytest.raises(ValueError):
        MODULE.validate_chain_url(url)
