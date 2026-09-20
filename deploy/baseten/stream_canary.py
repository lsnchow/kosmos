"""CPU-only Baseten Chain canary. No policy, world model, frames, or scores.

Checks real cross-Chainlet RPC and incremental NDJSON response delivery before
any model deployment. Receipt events must never enter a comparison frame list.
"""
import asyncio
import json
from typing import AsyncIterator

import truss_chains as chains
from pydantic import BaseModel


class CanaryReceipt(BaseModel):
    nonce: str
    index: int
    model_calls: int = 0
    scored: bool = False


class StreamCanaryReceipt(chains.ChainletBase):
    remote_config = chains.RemoteConfig(compute=chains.Compute(cpu_count=1, memory="2Gi"))

    async def run_remote(self, nonce: str, index: int) -> CanaryReceipt:
        if not nonce or len(nonce) > 128 or not 0 <= index <= 3:
            raise ValueError("Invalid canary request")
        return CanaryReceipt(nonce=nonce, index=index)


@chains.mark_entrypoint
class KosmosStreamCanary(chains.ChainletBase):
    remote_config = chains.RemoteConfig(compute=chains.Compute(cpu_count=1, memory="2Gi"))

    def __init__(self, receipt=chains.depends(StreamCanaryReceipt, retries=0)):
        self._receipt = receipt

    async def run_remote(self, nonce: str) -> AsyncIterator[str]:
        if not nonce or len(nonce) > 128:
            raise ValueError("A bounded nonce is required")
        yield json.dumps({"schema": "kosmos-stream-canary-v1", "type": "header", "nonce": nonce, "model_calls": 0}) + "\n"
        for index in range(3):
            result = await self._receipt.run_remote(nonce, index)
            yield json.dumps({"schema": "kosmos-stream-canary-v1", "type": "receipt", **result.model_dump()}) + "\n"
            await asyncio.sleep(0.25)
        yield json.dumps({"schema": "kosmos-stream-canary-v1", "type": "terminal", "nonce": nonce, "status": "completed_transport_only", "model_calls": 0}) + "\n"
