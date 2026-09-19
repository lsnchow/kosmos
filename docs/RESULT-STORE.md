# Immutable Chain result store

Real Baseten execution is disabled unless an explicitly configured immutable
S3-compatible result store is available. This avoids treating a callback as the
only copy of a terminal result.

The application stores an opaque 32-hex result key before the async POST. The
request contains only that key plus the run, episode, protocol, and unsigned
request-payload digest. It never contains a bucket, endpoint, presigned URL, or
credential. The Chain gets its bucket/prefix and SDK credentials solely from
deployment context, conditionally creates `<prefix>/<result-key>.json` before
returning a terminal result, and the application verifies the manifest digest,
size, and identities before recovery.

Set the `PLUMB_RESULT_STORE_*` configuration shown in `.env.example`, then
install the application optional dependency with `pip install '.[result-store]'`.
At Chain deployment time, set the three `PLUMB_RESULT_STORE_*_SECRET_NAME`
fields to the names of pre-created Baseten deployment secrets. The controller
gets values exclusively through `DeploymentContext.secrets` and maps those
values to boto3's explicit access-key, secret-key, and optional session-token
arguments. It does not assume a deployment secret becomes an environment
variable. Secret values are never supplied in a request or stored locally.

`S3ResultStore.preflight()` is the deliberately explicit reachability check. It
performs one `HeadBucket` only when an operator invokes it and returns a receipt
with the endpoint, bucket, timestamp, and provider request ID. Constructing the
store or registering the API makes no network call. A successful receipt proves
only that one configured store check succeeded; it is not a deployment,
qualification, or billing claim.

If writing or reading the manifest fails, the outcome stays failed or
unreconciled. PLUMB does not fall back to a callback-only result, synthesize a
completion, infer a store URL from Baseten routes, or automatically re-POST an
ambiguous request.
