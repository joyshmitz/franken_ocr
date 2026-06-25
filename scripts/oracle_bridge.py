#!/usr/bin/env python3
"""Test-only subprocess bridge to the pinned Unlimited-OCR oracle.

The shipping Rust binary must never link Python or torch. This script is only a
test harness helper: it launches a separate Python process for the pinned
reference stack when available, forces deterministic torch settings, and emits
structured JSON for parity checks. Its self-test is safe on machines without the
6.67 GB model or torch; missing/unpinned oracle dependencies are reported as a
skip-with-success, not as a false parity pass.
"""

from __future__ import annotations

import argparse
import enum
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any


BRIDGE_SCHEMA_VERSION = 1
PINNED_TORCH = "2.10.0"
PINNED_TRANSFORMERS = "4.57.1"
DEFAULT_SEED = 1337
DEFAULT_THREADS = 1
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ULP_TOLERANCE_BY_OP = {
    "matmul_f32": 4,
    "rmsnorm_f32": 2,
    "elementwise_f32": 2,
}
VALID_WORKER_RESULTS = frozenset({"pass", "fail", "skip_no_oracle", "skip_unpinned_oracle"})
EXPECTED_UNSET = object()


class EngineIdentity(str, enum.Enum):
    SUBJECT = "franken_ocr"
    ORACLE = "unlimited-ocr-oracle"


def emit(check: str, ok: bool, **fields: object) -> None:
    payload = {"check": check, "result": "pass" if ok else "fail", **fields}
    print(json.dumps(payload, sort_keys=True))


def json_response(result: str, **fields: object) -> dict[str, object]:
    return {"schema_version": BRIDGE_SCHEMA_VERSION, "result": result, **fields}


def assert_distinct_identities() -> bool:
    return EngineIdentity.SUBJECT.value != EngineIdentity.ORACLE.value


def identities_are_canonical() -> bool:
    return (
        EngineIdentity.SUBJECT.value == "franken_ocr"
        and EngineIdentity.ORACLE.value == "unlimited-ocr-oracle"
    )


def is_json_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_json_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_strict_true(value: object) -> bool:
    return isinstance(value, bool) and value


def is_non_empty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_json_int(payload: dict[str, Any], field: str, default: int) -> int:
    value = payload.get(field, default)
    if not is_json_int(value):
        raise ValueError(f"{field} must be an integer")
    return value


def validate_request_schema(payload: dict[str, Any]) -> None:
    schema_version = payload.get("schema_version")
    if not is_json_int(schema_version) or schema_version != BRIDGE_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {BRIDGE_SCHEMA_VERSION}")


def validate_request_identity(payload: dict[str, Any]) -> None:
    identity = payload.get("identity")
    if identity != EngineIdentity.SUBJECT.value:
        raise ValueError(f"identity must be {EngineIdentity.SUBJECT.value}")


def validate_worker_response_envelope(
    decoded: dict[str, object],
    *,
    expected_op: object = EXPECTED_UNSET,
    expected_output_len: int | None = None,
    expected_seed: int | None = None,
    expected_threads: int | None = None,
) -> dict[str, object]:
    schema_version = decoded.get("schema_version")
    if not is_json_int(schema_version) or schema_version != BRIDGE_SCHEMA_VERSION:
        return json_response(
            "fail",
            error=f"oracle worker response schema_version must be {BRIDGE_SCHEMA_VERSION}",
            worker_schema_version=schema_version,
        )
    result = decoded.get("result")
    if not isinstance(result, str) or result not in VALID_WORKER_RESULTS:
        return json_response("fail", error="oracle worker response result is invalid", worker_result=result)
    if result == "fail" and not is_non_empty_string(decoded.get("error")):
        return json_response("fail", error="oracle worker response error must be a non-empty string")
    if result in {"skip_no_oracle", "skip_unpinned_oracle"} and not is_non_empty_string(decoded.get("reason")):
        return json_response("fail", error="oracle worker response reason must be a non-empty string")
    if result == "pass":
        if decoded.get("identity") != EngineIdentity.ORACLE.value:
            return json_response(
                "fail",
                error=f"oracle worker response identity must be {EngineIdentity.ORACLE.value}",
                worker_identity=decoded.get("identity"),
            )
        op = decoded.get("op")
        if not is_non_empty_string(op):
            return json_response("fail", error="oracle worker response op must be a non-empty string")
        if expected_op is not EXPECTED_UNSET and op != expected_op:
            return json_response("fail", error="oracle worker response op mismatch", worker_op=op, expected_op=expected_op)
        seed = decoded.get("seed")
        if not is_json_int(seed):
            return json_response("fail", error="oracle worker response seed must be an integer")
        if expected_seed is not None and seed != expected_seed:
            return json_response(
                "fail",
                error="oracle worker response seed mismatch",
                worker_seed=seed,
                expected_seed=expected_seed,
            )
        if not is_strict_true(decoded.get("deterministic_algorithms")):
            return json_response("fail", error="oracle worker response deterministic_algorithms must be true")
        torch_threads = decoded.get("torch_threads")
        if not is_json_int(torch_threads) or torch_threads < 1:
            return json_response("fail", error="oracle worker response torch_threads must be a positive integer")
        if expected_threads is not None and torch_threads != expected_threads:
            return json_response(
                "fail",
                error="oracle worker response torch_threads mismatch",
                worker_threads=torch_threads,
                expected_threads=expected_threads,
            )
        determinism = decoded.get("determinism")
        if not isinstance(determinism, dict):
            return json_response("fail", error="oracle worker response determinism must be a JSON object")
        determinism_seed = determinism.get("seed")
        if not is_json_int(determinism_seed):
            return json_response("fail", error="oracle worker response determinism.seed must be an integer")
        if expected_seed is not None and determinism_seed != expected_seed:
            return json_response(
                "fail",
                error="oracle worker response determinism.seed mismatch",
                worker_seed=determinism_seed,
                expected_seed=expected_seed,
            )
        requested_threads = determinism.get("requested_threads")
        if not is_json_int(requested_threads) or requested_threads < 1:
            return json_response(
                "fail",
                error="oracle worker response determinism.requested_threads must be a positive integer",
            )
        determinism_threads = determinism.get("torch_threads")
        if not is_json_int(determinism_threads) or determinism_threads < 1:
            return json_response(
                "fail",
                error="oracle worker response determinism.torch_threads must be a positive integer",
            )
        if expected_threads is not None and (
            requested_threads != expected_threads or determinism_threads != expected_threads
        ):
            return json_response(
                "fail",
                error="oracle worker response determinism threads mismatch",
                requested_threads=requested_threads,
                torch_threads=determinism_threads,
                expected_threads=expected_threads,
            )
        if determinism.get("cublas_workspace_config") != DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG:
            return json_response(
                "fail",
                error="oracle worker response determinism cublas_workspace_config mismatch",
            )
        if expected_seed is not None and determinism.get("pythonhashseed") != str(expected_seed):
            return json_response(
                "fail",
                error="oracle worker response determinism pythonhashseed mismatch",
                pythonhashseed=determinism.get("pythonhashseed"),
                expected_pythonhashseed=str(expected_seed),
            )
        if not is_strict_true(determinism.get("transformers_set_seed")):
            return json_response("fail", error="oracle worker response determinism.transformers_set_seed must be true")
        if not is_strict_true(determinism.get("torch_manual_seed")):
            return json_response("fail", error="oracle worker response determinism.torch_manual_seed must be true")
        if not is_strict_true(determinism.get("torch_deterministic_algorithms")):
            return json_response(
                "fail",
                error="oracle worker response determinism.torch_deterministic_algorithms must be true",
            )
        output = decoded.get("output")
        if not isinstance(output, list):
            return json_response("fail", error="oracle worker response output must be a JSON array")
        if not output:
            return json_response("fail", error="oracle worker response output must be non-empty")
        if not all(is_json_number(value) for value in output):
            return json_response("fail", error="oracle worker response output must contain only finite JSON numbers")
        if expected_output_len is not None and len(output) != expected_output_len:
            return json_response(
                "fail",
                error="oracle worker response output length mismatch",
                worker_output_len=len(output),
                expected_output_len=expected_output_len,
            )
    return decoded


def parse_json_number_list(payload: dict[str, Any], field: str) -> list[float]:
    raw = payload[field]
    if not isinstance(raw, list):
        raise ValueError(f"{field} must be a JSON array")
    if not all(is_json_number(value) for value in raw):
        raise ValueError(f"{field} must contain only finite JSON numbers")
    return [float(value) for value in raw]


def reference_env(seed: int = DEFAULT_SEED, threads: int = DEFAULT_THREADS) -> dict[str, str]:
    if not is_json_int(seed):
        raise ValueError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if not is_json_int(threads):
        raise ValueError("threads must be an integer")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": str(seed),
            "CUBLAS_WORKSPACE_CONFIG": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "OMP_NUM_THREADS": str(threads),
            "TORCH_NUM_THREADS": str(threads),
            "FOCR_ORACLE_SEED": str(seed),
        }
    )
    return env


def parse_seed_threads(payload: dict[str, Any]) -> tuple[int, int]:
    seed = parse_json_int(payload, "seed", DEFAULT_SEED)
    threads = parse_json_int(payload, "threads", DEFAULT_THREADS)
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if threads < 1:
        raise ValueError("threads must be >= 1")
    return seed, threads


def f32_ordered_bits(value: float) -> int:
    bits = struct.unpack(">i", struct.pack(">f", float(value)))[0]
    return bits if bits >= 0 else 0x80000000 - bits


def round_f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", float(value)))[0]


def ulp_distance_f32(lhs: float, rhs: float) -> int:
    if math.isnan(lhs) or math.isnan(rhs):
        return 0 if math.isnan(lhs) and math.isnan(rhs) else 2**31
    if lhs == rhs:
        return 0
    return abs(f32_ordered_bits(lhs) - f32_ordered_bits(rhs))


def compare_vectors(lhs: list[float], rhs: list[float], max_ulp: int) -> dict[str, object]:
    if len(lhs) != len(rhs):
        return {"within_tolerance": False, "max_ulp": None, "error": "length mismatch"}
    distances = [ulp_distance_f32(a, b) for a, b in zip(lhs, rhs, strict=True)]
    observed = max(distances, default=0)
    return {"within_tolerance": observed <= max_ulp, "max_ulp": observed}


def subject_rmsnorm(values: list[float], weight: list[float], eps: float) -> list[float]:
    if len(values) != len(weight):
        raise ValueError("rmsnorm values and weight lengths differ")
    if not values:
        raise ValueError("rmsnorm input must be non-empty")
    square_sum = round_f32(0.0)
    for value in values:
        value_f32 = round_f32(value)
        square_sum = round_f32(square_sum + round_f32(value_f32 * value_f32))
    mean_square = round_f32(square_sum / round_f32(float(len(values))))
    inv_rms = round_f32(1.0 / math.sqrt(round_f32(mean_square + round_f32(eps))))
    return [
        round_f32(round_f32(round_f32(value) * inv_rms) * round_f32(scale))
        for value, scale in zip(values, weight, strict=True)
    ]


def parse_rmsnorm_payload(payload: dict[str, Any]) -> tuple[list[float], list[float], float, int, int]:
    seed, threads = parse_seed_threads(payload)
    values = parse_json_number_list(payload, "values")
    weight = parse_json_number_list(payload, "weight")
    eps_raw = payload["eps"]
    if not is_json_number(eps_raw):
        raise ValueError("eps must be a finite JSON number")
    eps = float(eps_raw)
    if eps < 0.0:
        raise ValueError("eps must be >= 0")
    if len(values) != len(weight):
        raise ValueError("rmsnorm values and weight lengths differ")
    if not values:
        raise ValueError("rmsnorm input must be non-empty")
    return values, weight, eps, seed, threads


def apply_reference_determinism(torch: Any, transformers: Any, seed: int, threads: int) -> dict[str, object]:
    record: dict[str, object] = {
        "seed": seed,
        "requested_threads": threads,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "transformers_set_seed": False,
        "torch_manual_seed": False,
        "torch_cuda_manual_seed_all": False,
        "torch_deterministic_algorithms": False,
        "torch_threads": None,
    }

    transformers.set_seed(seed)
    record["transformers_set_seed"] = True

    torch.manual_seed(seed)
    record["torch_manual_seed"] = True

    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        record["torch_cuda_manual_seed_all"] = True

    torch.use_deterministic_algorithms(True)
    record["torch_deterministic_algorithms"] = True

    torch.set_num_threads(threads)
    record["torch_threads"] = torch.get_num_threads()
    return record


def run_worker(payload: dict[str, Any]) -> dict[str, object]:
    try:
        validate_request_schema(payload)
        validate_request_identity(payload)
    except ValueError as exc:
        return json_response("fail", error=str(exc))
    op = payload.get("op")
    if op != "rmsnorm_f32":
        return json_response("fail", error=f"unsupported oracle op {op!r}")
    try:
        values_list, weight_list, eps, seed, threads = parse_rmsnorm_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return json_response("fail", error=f"invalid rmsnorm_f32 request: {exc}")

    try:
        import torch  # type: ignore[import-not-found]
        import transformers  # type: ignore[import-not-found]
    except ImportError as exc:
        return json_response("skip_no_oracle", reason=f"missing oracle dependency: {exc.name}")

    torch_version = torch.__version__.split("+", 1)[0]
    transformers_version = transformers.__version__
    if torch_version != PINNED_TORCH or transformers_version != PINNED_TRANSFORMERS:
        return json_response(
            "skip_unpinned_oracle",
            reason="oracle dependency versions are not pinned",
            torch_version=torch.__version__,
            transformers_version=transformers_version,
            required_torch=PINNED_TORCH,
            required_transformers=PINNED_TRANSFORMERS,
        )

    determinism = apply_reference_determinism(torch, transformers, seed, threads)

    values = torch.tensor(values_list, dtype=torch.float32)
    weight = torch.tensor(weight_list, dtype=torch.float32)
    out = values * torch.rsqrt(torch.mean(values * values) + eps) * weight
    return json_response(
        "pass",
        identity=EngineIdentity.ORACLE.value,
        op=op,
        seed=seed,
        determinism=determinism,
        deterministic_algorithms=determinism["torch_deterministic_algorithms"],
        torch_threads=determinism["torch_threads"],
        output=[float(v) for v in out.tolist()],
    )


def call_oracle(payload: dict[str, Any], python: str = sys.executable, timeout_s: float = 10.0) -> dict[str, object]:
    try:
        validate_request_schema(payload)
        validate_request_identity(payload)
        seed, threads = parse_seed_threads(payload)
    except ValueError as exc:
        return json_response("fail", error=str(exc))
    op = payload.get("op")
    if op != "rmsnorm_f32":
        return json_response("fail", error=f"unsupported oracle op {op!r}")
    try:
        values_list, _, _, _, _ = parse_rmsnorm_payload(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return json_response("fail", error=f"invalid rmsnorm_f32 request: {exc}")
    expected_output_len = len(values_list)
    try:
        proc = subprocess.run(
            [python, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload, sort_keys=True),
            text=True,
            capture_output=True,
            env=reference_env(seed, threads),
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return json_response("fail", error="oracle worker timed out", timeout_s=timeout_s)
    if proc.returncode != 0:
        return json_response("fail", error="oracle worker failed", returncode=proc.returncode, stderr=proc.stderr)
    try:
        decoded = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return json_response("fail", error=f"oracle worker emitted invalid JSON: {exc}", stdout=proc.stdout)
    if not isinstance(decoded, dict):
        return json_response("fail", error="oracle worker did not return a JSON object")
    return validate_worker_response_envelope(
        decoded,
        expected_op=op,
        expected_output_len=expected_output_len,
        expected_seed=seed,
        expected_threads=threads,
    )


def self_test() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, **fields: object) -> None:
        emit(name, cond, **fields)
        if not cond:
            failures.append(name)

    check("engine-identities-distinct", assert_distinct_identities())
    check("engine-identities-canonical", identities_are_canonical())
    check("ulp-tolerance-matmul", ULP_TOLERANCE_BY_OP["matmul_f32"] == 4)
    check("ulp-tolerance-elementwise", ULP_TOLERANCE_BY_OP["elementwise_f32"] == 2)
    check("ulp-tolerance-rmsnorm", ULP_TOLERANCE_BY_OP["rmsnorm_f32"] == 2)
    env = reference_env(seed=7, threads=2)
    check(
        "deterministic-reference-env",
        env["PYTHONHASHSEED"] == "7"
        and env["CUBLAS_WORKSPACE_CONFIG"] == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        and env["OMP_NUM_THREADS"] == "2"
        and env["TORCH_NUM_THREADS"] == "2"
        and env["FOCR_ORACLE_SEED"] == "7",
    )
    try:
        reference_env(seed=-1)
    except ValueError as exc:
        check("reference-env-rejects-negative-seed", "seed must be >= 0" in str(exc))
    else:
        check("reference-env-rejects-negative-seed", False)
    try:
        reference_env(seed=True)
    except ValueError as exc:
        check("reference-env-rejects-bool-seed", "seed must be an integer" in str(exc))
    else:
        check("reference-env-rejects-bool-seed", False)
    try:
        reference_env(threads=False)
    except ValueError as exc:
        check("reference-env-rejects-bool-threads", "threads must be an integer" in str(exc))
    else:
        check("reference-env-rejects-bool-threads", False)

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()

        def __init__(self) -> None:
            self.seed: int | None = None
            self.threads = 0
            self.deterministic = False

        def manual_seed(self, seed: int) -> None:
            self.seed = seed

        def use_deterministic_algorithms(self, enabled: bool) -> None:
            self.deterministic = enabled

        def set_num_threads(self, threads: int) -> None:
            self.threads = threads

        def get_num_threads(self) -> int:
            return self.threads

    class FakeTransformers:
        def __init__(self) -> None:
            self.seed: int | None = None

        def set_seed(self, seed: int) -> None:
            self.seed = seed

    old_hashseed = os.environ.get("PYTHONHASHSEED")
    old_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    try:
        os.environ["PYTHONHASHSEED"] = "11"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        fake_torch = FakeTorch()
        fake_transformers = FakeTransformers()
        record = apply_reference_determinism(fake_torch, fake_transformers, seed=11, threads=3)
    finally:
        if old_hashseed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = old_hashseed
        if old_cublas is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = old_cublas
    check(
        "determinism-record-self-test",
        record["seed"] == 11
        and record["requested_threads"] == 3
        and record["pythonhashseed"] == "11"
        and record["cublas_workspace_config"] == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG
        and is_strict_true(record["transformers_set_seed"])
        and is_strict_true(record["torch_manual_seed"])
        and is_strict_true(record["torch_deterministic_algorithms"])
        and record["torch_threads"] == 3,
        detail=record,
    )

    request = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "op": "rmsnorm_f32",
        "identity": EngineIdentity.SUBJECT.value,
        "values": [1.0, -2.0, 3.5, -4.25],
        "weight": [1.0, 0.5, 1.25, 0.75],
        "eps": 1e-6,
        "seed": DEFAULT_SEED,
        "threads": DEFAULT_THREADS,
    }
    subject = subject_rmsnorm(request["values"], request["weight"], request["eps"])
    local_cmp = compare_vectors(subject, subject, ULP_TOLERANCE_BY_OP["rmsnorm_f32"])
    check("subject-rmsnorm-self-compare", bool(local_cmp["within_tolerance"]), max_ulp=local_cmp["max_ulp"])

    def oracle_pass_response(
        *,
        op: str = "rmsnorm_f32",
        seed: int = DEFAULT_SEED,
        threads: int = DEFAULT_THREADS,
        output: list[object] | None = None,
    ) -> dict[str, object]:
        determinism = {
            "seed": seed,
            "requested_threads": threads,
            "pythonhashseed": str(seed),
            "cublas_workspace_config": DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
            "transformers_set_seed": True,
            "torch_manual_seed": True,
            "torch_cuda_manual_seed_all": False,
            "torch_deterministic_algorithms": True,
            "torch_threads": threads,
        }
        return json_response(
            "pass",
            identity=EngineIdentity.ORACLE.value,
            op=op,
            seed=seed,
            determinism=determinism,
            deterministic_algorithms=True,
            torch_threads=threads,
            output=[1.0] if output is None else output,
        )

    valid_skip_response = json_response("skip_no_oracle", reason="missing torch")
    check(
        "oracle-response-envelope-accepts-valid-skip",
        validate_worker_response_envelope(valid_skip_response) is valid_skip_response,
        detail=valid_skip_response,
    )
    valid_fail_response = json_response("fail", error="worker failed cleanly")
    check(
        "oracle-response-envelope-accepts-valid-fail",
        validate_worker_response_envelope(valid_fail_response) is valid_fail_response,
        detail=valid_fail_response,
    )
    valid_pass_response = oracle_pass_response(output=[1.0, 2.0])
    check(
        "oracle-response-envelope-accepts-oracle-pass",
        validate_worker_response_envelope(
            valid_pass_response,
            expected_op="rmsnorm_f32",
            expected_output_len=2,
            expected_seed=DEFAULT_SEED,
            expected_threads=DEFAULT_THREADS,
        )
        is valid_pass_response,
        detail=valid_pass_response,
    )
    missing_response_schema = {"result": "pass", "identity": EngineIdentity.ORACLE.value}
    missing_response_schema_result = validate_worker_response_envelope(missing_response_schema)
    check(
        "oracle-response-envelope-rejects-missing-schema",
        missing_response_schema_result.get("result") == "fail"
        and "response schema_version must be 1" in str(missing_response_schema_result.get("error")),
        detail=missing_response_schema_result,
    )
    bool_response_schema = json_response("pass", identity=EngineIdentity.ORACLE.value)
    bool_response_schema["schema_version"] = True
    bool_response_schema_result = validate_worker_response_envelope(bool_response_schema)
    check(
        "oracle-response-envelope-rejects-bool-schema",
        bool_response_schema_result.get("result") == "fail"
        and "response schema_version must be 1" in str(bool_response_schema_result.get("error")),
        detail=bool_response_schema_result,
    )
    invalid_response_result = json_response("maybe", identity=EngineIdentity.ORACLE.value)
    invalid_response_result_check = validate_worker_response_envelope(invalid_response_result)
    check(
        "oracle-response-envelope-rejects-invalid-result",
        invalid_response_result_check.get("result") == "fail"
        and "response result is invalid" in str(invalid_response_result_check.get("error")),
        detail=invalid_response_result_check,
    )
    subject_pass_response = oracle_pass_response()
    subject_pass_response["identity"] = EngineIdentity.SUBJECT.value
    subject_pass_response_result = validate_worker_response_envelope(subject_pass_response)
    check(
        "oracle-response-envelope-rejects-subject-pass",
        subject_pass_response_result.get("result") == "fail"
        and "response identity must be unlimited-ocr-oracle" in str(subject_pass_response_result.get("error")),
        detail=subject_pass_response_result,
    )
    missing_reason_response = json_response("skip_no_oracle")
    missing_reason_response_result = validate_worker_response_envelope(missing_reason_response)
    check(
        "oracle-response-envelope-rejects-skip-without-reason",
        missing_reason_response_result.get("result") == "fail"
        and "response reason must be a non-empty string" in str(missing_reason_response_result.get("error")),
        detail=missing_reason_response_result,
    )
    missing_error_response = json_response("fail")
    missing_error_response_result = validate_worker_response_envelope(missing_error_response)
    check(
        "oracle-response-envelope-rejects-fail-without-error",
        missing_error_response_result.get("result") == "fail"
        and "response error must be a non-empty string" in str(missing_error_response_result.get("error")),
        detail=missing_error_response_result,
    )
    missing_output_response = oracle_pass_response()
    missing_output_response.pop("output")
    missing_output_response_result = validate_worker_response_envelope(missing_output_response)
    check(
        "oracle-response-envelope-rejects-pass-without-output",
        missing_output_response_result.get("result") == "fail"
        and "response output must be a JSON array" in str(missing_output_response_result.get("error")),
        detail=missing_output_response_result,
    )
    empty_output_response = oracle_pass_response(output=[])
    empty_output_response_result = validate_worker_response_envelope(empty_output_response)
    check(
        "oracle-response-envelope-rejects-empty-output",
        empty_output_response_result.get("result") == "fail"
        and "response output must be non-empty" in str(empty_output_response_result.get("error")),
        detail=empty_output_response_result,
    )
    bool_output_response = oracle_pass_response(output=[True])
    bool_output_response_result = validate_worker_response_envelope(bool_output_response)
    check(
        "oracle-response-envelope-rejects-bool-output",
        bool_output_response_result.get("result") == "fail"
        and "response output must contain only finite JSON numbers" in str(bool_output_response_result.get("error")),
        detail=bool_output_response_result,
    )
    nonfinite_output_response = oracle_pass_response(output=[float("inf")])
    nonfinite_output_response_result = validate_worker_response_envelope(nonfinite_output_response)
    check(
        "oracle-response-envelope-rejects-nonfinite-output",
        nonfinite_output_response_result.get("result") == "fail"
        and "response output must contain only finite JSON numbers" in str(nonfinite_output_response_result.get("error")),
        detail=nonfinite_output_response_result,
    )
    missing_op_response = oracle_pass_response()
    missing_op_response.pop("op")
    missing_op_response_result = validate_worker_response_envelope(missing_op_response)
    check(
        "oracle-response-envelope-rejects-pass-without-op",
        missing_op_response_result.get("result") == "fail"
        and "response op must be a non-empty string" in str(missing_op_response_result.get("error")),
        detail=missing_op_response_result,
    )
    wrong_op_response = oracle_pass_response(op="matmul_f32")
    wrong_op_response_result = validate_worker_response_envelope(wrong_op_response, expected_op="rmsnorm_f32")
    check(
        "oracle-response-envelope-rejects-wrong-op",
        wrong_op_response_result.get("result") == "fail"
        and "response op mismatch" in str(wrong_op_response_result.get("error")),
        detail=wrong_op_response_result,
    )
    wrong_output_len_response = oracle_pass_response(output=[1.0])
    wrong_output_len_result = validate_worker_response_envelope(wrong_output_len_response, expected_output_len=2)
    check(
        "oracle-response-envelope-rejects-wrong-output-len",
        wrong_output_len_result.get("result") == "fail"
        and "response output length mismatch" in str(wrong_output_len_result.get("error")),
        detail=wrong_output_len_result,
    )
    missing_determinism_response = oracle_pass_response()
    missing_determinism_response.pop("determinism")
    missing_determinism_result = validate_worker_response_envelope(missing_determinism_response)
    check(
        "oracle-response-envelope-rejects-missing-determinism",
        missing_determinism_result.get("result") == "fail"
        and "response determinism must be a JSON object" in str(missing_determinism_result.get("error")),
        detail=missing_determinism_result,
    )
    wrong_seed_response = oracle_pass_response(seed=DEFAULT_SEED + 1)
    wrong_seed_result = validate_worker_response_envelope(wrong_seed_response, expected_seed=DEFAULT_SEED)
    check(
        "oracle-response-envelope-rejects-wrong-seed",
        wrong_seed_result.get("result") == "fail"
        and "response seed mismatch" in str(wrong_seed_result.get("error")),
        detail=wrong_seed_result,
    )
    wrong_hashseed_response = oracle_pass_response()
    wrong_hashseed_determinism = wrong_hashseed_response["determinism"]
    if isinstance(wrong_hashseed_determinism, dict):
        wrong_hashseed_determinism["pythonhashseed"] = str(DEFAULT_SEED + 1)
    wrong_hashseed_result = validate_worker_response_envelope(wrong_hashseed_response, expected_seed=DEFAULT_SEED)
    check(
        "oracle-response-envelope-rejects-wrong-pythonhashseed",
        wrong_hashseed_result.get("result") == "fail"
        and "response determinism pythonhashseed mismatch" in str(wrong_hashseed_result.get("error")),
        detail=wrong_hashseed_result,
    )
    wrong_threads_response = oracle_pass_response(threads=DEFAULT_THREADS + 1)
    wrong_threads_result = validate_worker_response_envelope(wrong_threads_response, expected_threads=DEFAULT_THREADS)
    check(
        "oracle-response-envelope-rejects-wrong-threads",
        wrong_threads_result.get("result") == "fail"
        and "response torch_threads mismatch" in str(wrong_threads_result.get("error")),
        detail=wrong_threads_result,
    )

    missing_schema_request = dict(request)
    missing_schema_request.pop("schema_version")
    missing_schema = run_worker(missing_schema_request)
    check(
        "oracle-worker-rejects-missing-schema",
        missing_schema.get("result") == "fail" and "schema_version must be 1" in str(missing_schema.get("error")),
        detail=missing_schema,
    )
    wrong_schema_request = dict(request)
    wrong_schema_request["schema_version"] = 2
    wrong_schema = call_oracle(wrong_schema_request)
    check(
        "oracle-call-rejects-wrong-schema",
        wrong_schema.get("result") == "fail" and "schema_version must be 1" in str(wrong_schema.get("error")),
        detail=wrong_schema,
    )
    bool_schema_request = dict(request)
    bool_schema_request["schema_version"] = True
    bool_schema = run_worker(bool_schema_request)
    check(
        "oracle-worker-rejects-bool-schema",
        bool_schema.get("result") == "fail" and "schema_version must be 1" in str(bool_schema.get("error")),
        detail=bool_schema,
    )
    missing_identity_request = dict(request)
    missing_identity_request.pop("identity")
    missing_identity = run_worker(missing_identity_request)
    check(
        "oracle-worker-rejects-missing-identity",
        missing_identity.get("result") == "fail" and "identity must be franken_ocr" in str(missing_identity.get("error")),
        detail=missing_identity,
    )
    oracle_identity_request = dict(request)
    oracle_identity_request["identity"] = EngineIdentity.ORACLE.value
    oracle_identity = call_oracle(oracle_identity_request)
    check(
        "oracle-call-rejects-oracle-identity",
        oracle_identity.get("result") == "fail" and "identity must be franken_ocr" in str(oracle_identity.get("error")),
        detail=oracle_identity,
    )
    bool_identity_request = dict(request)
    bool_identity_request["identity"] = True
    bool_identity = run_worker(bool_identity_request)
    check(
        "oracle-worker-rejects-bool-identity",
        bool_identity.get("result") == "fail" and "identity must be franken_ocr" in str(bool_identity.get("error")),
        detail=bool_identity,
    )
    unsupported_op_request = dict(request)
    unsupported_op_request["op"] = "matmul_f32"
    unsupported_op = call_oracle(unsupported_op_request)
    check(
        "oracle-call-rejects-unsupported-op-before-spawn",
        unsupported_op.get("result") == "fail" and "unsupported oracle op" in str(unsupported_op.get("error")),
        detail=unsupported_op,
    )

    negative_seed_request = dict(request)
    negative_seed_request["seed"] = -1
    negative_seed = call_oracle(negative_seed_request)
    check(
        "oracle-call-rejects-negative-seed",
        negative_seed.get("result") == "fail" and "seed must be >= 0" in str(negative_seed.get("error")),
        detail=negative_seed,
    )
    bool_seed_request = dict(request)
    bool_seed_request["seed"] = True
    bool_seed = call_oracle(bool_seed_request)
    check(
        "oracle-call-rejects-bool-seed",
        bool_seed.get("result") == "fail" and "seed must be an integer" in str(bool_seed.get("error")),
        detail=bool_seed,
    )
    bool_values_request = dict(request)
    bool_values_request["values"] = [True, -2.0, 3.5, -4.25]
    bool_values = run_worker(bool_values_request)
    check(
        "oracle-worker-rejects-bool-values",
        bool_values.get("result") == "fail" and "values must contain only finite JSON numbers" in str(bool_values.get("error")),
        detail=bool_values,
    )
    bool_values_call = call_oracle(bool_values_request)
    check(
        "oracle-call-rejects-bool-values-before-spawn",
        bool_values_call.get("result") == "fail"
        and "values must contain only finite JSON numbers" in str(bool_values_call.get("error")),
        detail=bool_values_call,
    )
    nonfinite_eps_request = dict(request)
    nonfinite_eps_request["eps"] = float("nan")
    nonfinite_eps = run_worker(nonfinite_eps_request)
    check(
        "oracle-worker-rejects-nonfinite-eps",
        nonfinite_eps.get("result") == "fail" and "eps must be a finite JSON number" in str(nonfinite_eps.get("error")),
        detail=nonfinite_eps,
    )

    oracle = call_oracle(request)
    if oracle.get("result") in {"skip_no_oracle", "skip_unpinned_oracle"}:
        emit("oracle-subprocess-smoke", True, skipped=True, reason=oracle.get("reason"), detail=oracle)
    else:
        check("oracle-subprocess-result", oracle.get("result") == "pass", detail=oracle)
        if oracle.get("result") == "pass":
            output = oracle.get("output")
            check("oracle-output-vector", isinstance(output, list), detail=oracle)
            if isinstance(output, list):
                try:
                    output_values = [float(value) for value in output]
                except (TypeError, ValueError) as exc:
                    check("oracle-output-values-numeric", False, error=str(exc), detail=oracle)
                else:
                    cmp = compare_vectors(subject, output_values, ULP_TOLERANCE_BY_OP["rmsnorm_f32"])
                    check(
                        "oracle-rmsnorm-within-ulp",
                        bool(cmp["within_tolerance"]),
                        max_ulp=cmp["max_ulp"],
                        tolerance=ULP_TOLERANCE_BY_OP["rmsnorm_f32"],
                    )
            check("oracle-identity", oracle.get("identity") == EngineIdentity.ORACLE.value)
            check("oracle-deterministic-flag", is_strict_true(oracle.get("deterministic_algorithms")))
            determinism = oracle.get("determinism")
            check("oracle-determinism-record", isinstance(determinism, dict), detail=oracle)
            if isinstance(determinism, dict):
                check("oracle-determinism-seed", determinism.get("seed") == DEFAULT_SEED, detail=determinism)
                check("oracle-determinism-threads", determinism.get("torch_threads") == DEFAULT_THREADS, detail=determinism)
                check(
                    "oracle-determinism-cublas",
                    determinism.get("cublas_workspace_config") == DETERMINISTIC_CUBLAS_WORKSPACE_CONFIG,
                    detail=determinism,
                )
                check("oracle-determinism-manual-seed", is_strict_true(determinism.get("torch_manual_seed")), detail=determinism)

    if failures:
        emit("oracle-bridge-self-test", False, failed=failures)
        return 1
    emit("oracle-bridge-self-test", True, checks_passed=True)
    return 0


def worker_main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        print(json.dumps(json_response("fail", error=f"invalid request JSON: {exc}"), sort_keys=True))
        return 0
    if not isinstance(payload, dict):
        print(json.dumps(json_response("fail", error="request must be a JSON object"), sort_keys=True))
        return 0
    print(json.dumps(run_worker(payload), sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="run stdlib-safe bridge self-tests")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        return worker_main()
    if args.self_test:
        return self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
