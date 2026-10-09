"""STAGED ONLY: one submission on an already-owned Tinker 0.31.0 sampler.

The caller owns admission/accounting, the sampler, its service close, and raw
SampleResponse persistence before parse/accounting. This module never creates
a ServiceClient, retries a sample, cancels a future, or changes model arguments.
Private SDK APIs below are version/source-pinned, not a public compatibility API.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


SDK_PINS = {
    "lib/public_interfaces/sampling_client.py": "fa246d5870bece5144c075e4ebb495a8e1fd5dadf0cebc6a9047163ea9acce97",
    "lib/api_future_impl.py": "aebd8e2eddb0e0e3c4a5fcb9df1aa1bb4b7bed9b4bec8374bb494931566fc943",
    "resources/sampling.py": "7e67039fbf593f8990c6230f7cfb72f396bc657cc580e0a51b889a04c341a5a3",
    "types/shared/untyped_api_future.py": "c747a330cedcfc913a8a08175498b713a013de4991bfd2ab95fb7357d93336ba",
    "types/sample_request.py": "46ac0281cde94597f3d262f0f73331d5fe249779c2b4de790a8183ce122431dc",
    "lib/public_interfaces/api_future.py": "c1458f4648b3556857a916ee413f06dda9f0cd42953e85d89adc7e37683a0967",
    "lib/internal_client_holder.py": "78bfe7a113f5b36191fd9e72f715e98b86e8fd74d125fcdf612732109e9d5492",
    "lib/client_connection_pool_type.py": "9030732ac6805c57c0cae6f4f772a7d3ac717c10f9e8b7596d207d95c49b12ae",
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def write_new(path: Path, value: Any) -> dict:
    """Exclusive, flushed/fsynced structured bytes; never overwrite evidence."""
    raw = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


@dataclass(frozen=True)
class Bindings:
    sample_request: Any
    sample_response: Any
    sample_pool: Any
    future_factory: Callable
    attach_sequence_ids: Callable
    check_returned: Callable
    check_target: Callable
    target_to_model: Callable
    check_alt: Callable
    source_refs: tuple
    synthetic: bool = False


def installed_bindings() -> Bindings:
    """Import-only; authenticate the installed SDK before using private APIs."""
    if importlib.metadata.version("tinker") != "0.31.0":
        raise ValueError("SDK_VERSION_NOT_REVIEWED")
    spec = importlib.util.find_spec("tinker")
    if spec is None or spec.origin is None:
        raise ValueError("SDK_SOURCE_NOT_FOUND")
    root = Path(spec.origin).parent
    refs = []
    for relative, digest in SDK_PINS.items():
        path = root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError("SDK_SOURCE_NOT_REVIEWED")
        refs.append({"path": str(path.resolve()), "sha256": actual})
    from tinker import types
    from tinker.lib.api_future_impl import _APIFuture
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType
    from tinker.lib.public_interfaces.sampling_client import (
        _attach_sequence_ids, _check_prompt_alt_tokens_returned,
        _check_target_prompt_logprobs, _tensor_data_to_model,
        _check_prompt_alt_tokens_k,
    )
    return Bindings(types.SampleRequest, types.SampleResponse,
                    ClientConnectionPoolType.SAMPLE, _APIFuture,
                    _attach_sequence_ids, _check_prompt_alt_tokens_returned,
                    _check_target_prompt_logprobs, _tensor_data_to_model,
                    _check_prompt_alt_tokens_k, tuple(refs))


def sanitized_error(error: BaseException) -> dict:
    """Never read message/str/repr, headers, request, response, body, or kwargs."""
    def class_name(value: BaseException) -> str:
        name = type(value).__name__
        return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,95}", name) else "Exception"
    classes, seen = [], set()
    current = error
    for _ in range(4):
        if id(current) in seen:
            break
        seen.add(id(current))
        classes.append(class_name(current))
        current = current.__cause__ or current.__context__
        if current is None:
            break
    status = getattr(error, "status_code", None)
    if type(status) is not int or not 100 <= status <= 599:
        status = None
    return {"exception_class": classes[0], "cause_classes": classes[1:],
            "http_status_code": status, "exception_text_serialized": False,
            "headers_serialized": False, "body_serialized": False}


class CapturedSamplingFailure(RuntimeError):
    """Safe public exception; original provider text is deliberately suppressed."""
    def __init__(self, phase: str, directory: Path, diagnostic_ref: dict | None):
        super().__init__("CAPTURED_SAMPLING_FAILURE:" + phase)
        self.phase = phase
        self.directory = directory
        self.diagnostic_ref = diagnostic_ref


class CapturedCall:
    """Sync bounded observation; a timeout neither resubmits nor cancels work.

    Use once and only off the holder's event-loop thread. Accepted work can
    remain live after a timeout; the caller must retain its unknown obligation
    and perform owned cleanup. Reopening a directory never resubmits it.
    """
    def __init__(self, sampler, bindings, directory, caller_request_sha256):
        self.sampler = sampler
        self.bindings = bindings
        self.directory = directory
        self.caller_request_sha256 = caller_request_sha256
        self.dispatch_future = None
        self.response_future = None
        self.ack = None
        self.identity = None
        self.waited = False

    def fail(self, phase: str, error: BaseException) -> CapturedSamplingFailure:
        diagnostic = {"kind": "SANITIZED_SAMPLING_TRANSPORT_FAILURE_V1",
                      "phase": phase, "synthetic": self.bindings.synthetic,
                      "caller_request_sha256": self.caller_request_sha256,
                      "recovery_identity": self.identity,
                      "provider_outcome": "UNKNOWN_NOT_COMPLETED_BY_THIS_WRAPPER",
                      "sample_resubmitted": False, "future_cancel_requested": False,
                      **sanitized_error(error)}
        ref = None
        try:
            ref = write_new(self.directory / ("failure-" + phase + ".json"), diagnostic)
        except BaseException:
            # A capture failure is not authority to retry a sample or replace
            # evidence. Existing intent/ack remain; caller keeps the hold.
            pass
        return CapturedSamplingFailure(phase, self.directory, ref)

    def result(self, timeout: float = 50.0):
        if self.waited:
            raise ValueError("ONE_OBSERVATION_ONLY_NO_REPLAY")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("FINITE_POSITIVE_OBSERVATION_TIMEOUT_REQUIRED")
        self.waited = True
        deadline = time.monotonic() + timeout
        try:
            self.response_future = self.dispatch_future.result(timeout=timeout)
        except CapturedSamplingFailure:
            raise
        except BaseException as error:
            raise self.fail("ACK_WAIT_OBSERVATION", error) from None
        try:
            # _APIFuture.result uses a concurrent future timeout without the
            # async wait_for cancellation path. It may already be polling.
            response = self.response_future.result(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException as error:
            raise self.fail("RESPONSE_WAIT", error) from None
        try:
            self.bindings.check_returned(response, self.prompt_alt_tokens_k)
            # Exactly the SDK's submission-time ID stamping; tokens unchanged.
            return self.bindings.attach_sequence_ids(response, self.ack.sample_sequence_ids)
        except BaseException as error:
            raise self.fail("RESPONSE_METADATA", error) from None


def start_captured_sample(sampler, *, prompt, num_samples, sampling_params,
                          directory: Path, caller_request_sha256: str,
                          include_prompt_logprobs=False, topk_prompt_logprobs=0,
                          topk_sample_logprobs=0, target_prompt_logprobs=None,
                          prompt_alt_tokens_k=0, bindings: Bindings | None = None) -> CapturedCall:
    """Schedule one official asample on the existing sampler/session/holder.

    Caller must claim the exact request and own the full unchanged conversation
    before invoking this function. Sampler use must be serialized by the caller.
    Optional SDK tensor checks/conversion match SamplingClient.sample exactly.
    This is a staged opt-in interface, not monkey-patching SamplingClient.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", caller_request_sha256):
        raise ValueError("AUTHENTIC_CALLER_REQUEST_DIGEST_REQUIRED")
    bindings = bindings or installed_bindings()
    target_model = None
    if target_prompt_logprobs is not None:
        bindings.check_target(target_prompt_logprobs)
        target_model = bindings.target_to_model(target_prompt_logprobs)
    bindings.check_alt(prompt_alt_tokens_k)
    directory = Path(directory)
    directory.mkdir(parents=False, exist_ok=False)
    call = CapturedCall(sampler, bindings, directory, caller_request_sha256)
    call.prompt_alt_tokens_k = prompt_alt_tokens_k
    write_new(directory / "intent.json", {
        "kind": "ONE_SUBMISSION_CAPTURE_INTENT_V1", "synthetic": bindings.synthetic,
        "caller_request_sha256": caller_request_sha256, "sdk_sources": bindings.source_refs,
        "maximum_sample_submissions": 1, "service_created": False,
        "sampler_created": False, "sample_retry_enabled": False,
    })

    async def dispatch():
        phase = "LOCAL_REQUEST_PREPARATION"
        try:
            holder = sampler.holder
            seq_id = sampler._request_id_counter
            sampler._request_id_counter += 1  # Same original counter, on original loop.
            request = bindings.sample_request(
                sampling_session_id=sampler._sampling_session_id, seq_id=seq_id,
                num_samples=num_samples, prompt=prompt, sampling_params=sampling_params,
                prompt_logprobs=include_prompt_logprobs,
                topk_prompt_logprobs=topk_prompt_logprobs,
                topk_sample_logprobs=topk_sample_logprobs,
                target_prompt_logprobs=target_model, prompt_alt_tokens_k=prompt_alt_tokens_k,
                record_stability_info=sampler._record_stability_info,
            )
            # SDK's resource serialization mode; keep only its digest locally.
            wire = request.model_dump(exclude_unset=False, exclude_none=True, mode="json")
            request_ref = write_new(directory / "request-identity.json", {
                "kind": "SDK_PARSED_REQUEST_IDENTITY_V1", "synthetic": bindings.synthetic,
                "sampling_session_id": sampler._sampling_session_id, "seq_id": seq_id,
                "caller_request_sha256": caller_request_sha256,
                "sdk_request_canonical_sha256": hashlib.sha256(canonical(wire)).hexdigest(),
                "request_tokens_or_logprobs_serialized": False,
            })
            estimated = holder.estimate_bytes_count_in_model_input(prompt)
            phase = "ASAMPLE_DISPATCH_OR_ACK"
            async with holder.sample_dispatch_rate_limit(estimated):
                with holder.aclient(bindings.sample_pool) as client:
                    ack = await client.sampling.asample(
                        request=request, max_retries=0,
                        extra_headers={"X-Tinker-Sampling-Backpressure": "1"},
                    )
            phase = "ACK_CAPTURE"
            call.ack = ack
            # Explicit schema fields only. Never dump unknown extras or headers.
            call.identity = {"sampling_session_id": sampler._sampling_session_id,
                             "seq_id": seq_id, "request_id": ack.request_id,
                             "model_id": ack.model_id,
                             "sample_sequence_ids": ack.sample_sequence_ids}
            write_new(directory / "acknowledgement.json", {
                "kind": "SDK_PARSED_ASAMPLE_ACK_V1", "synthetic": bindings.synthetic,
                "representation": "ORIGINAL_SDK_SCHEMA_FIELDS_NOT_HTTP_WIRE_BYTES",
                "request_identity": request_ref, **call.identity,
            })
            # Constructor begins polling. The durable acknowledgement MUST precede it.
            phase = "FUTURE_CONSTRUCTION_AFTER_DURABLE_ACK"
            return bindings.future_factory(
                bindings.sample_response, holder, ack, request_start_time=time.time(),
                request_type="Sample", queue_state_observer=sampler,
                futures_poller=sampler._get_futures_poller(),
            )
        except BaseException as error:
            raise call.fail(phase, error) from None

    coroutine = dispatch()
    try:
        call.dispatch_future = sampler.holder.run_coroutine_threadsafe(coroutine)
    except BaseException as error:
        coroutine.close()
        raise call.fail("LOCAL_SCHEDULING", error) from None
    return call
