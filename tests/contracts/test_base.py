from __future__ import annotations

import json
from typing import Literal

import pytest
from pydantic import ValidationError

from huyawo_adapt.contracts import ContractModel


class ExampleContract(ContractModel):
    contract_type: Literal["example"] = "example"
    name: str
    count: int


class FloatContract(ContractModel):
    contract_type: Literal["float"] = "float"
    value: float


def test_canonical_json_is_sorted_compact_and_utf8_safe() -> None:
    contract = ExampleContract(name="café", count=2)

    assert contract.canonical_json() == (
        '{"contract_type":"example","count":2,"name":"café","schema_version":"1.0"}'
    )
    assert contract.canonical_bytes() == contract.canonical_json().encode("utf-8")


def test_fingerprint_is_stable_for_equivalent_instances() -> None:
    first = ExampleContract(name="alpha", count=3)
    second = ExampleContract(count=3, name="alpha")

    assert first.fingerprint() == second.fingerprint()
    assert len(first.fingerprint()) == 64


def test_round_trip_preserves_contract_and_fingerprint() -> None:
    original = ExampleContract(name="alpha", count=3)
    restored = ExampleContract.from_canonical_json(original.canonical_json())

    assert restored == original
    assert restored.fingerprint() == original.fingerprint()


def test_contracts_are_frozen() -> None:
    contract = ExampleContract(name="alpha", count=3)

    with pytest.raises(ValidationError):
        contract.name = "beta"


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ExampleContract.model_validate(
            {
                "contract_type": "example",
                "schema_version": "1.0",
                "name": "alpha",
                "count": 3,
                "unexpected": True,
            }
        )


def test_strict_validation_rejects_type_coercion() -> None:
    with pytest.raises(ValidationError):
        ExampleContract.model_validate(
            {
                "name": "alpha",
                "count": "3",
            }
        )


def test_non_finite_float_is_rejected() -> None:
    with pytest.raises(ValidationError):
        FloatContract(value=float("nan"))


def test_unknown_schema_version_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ExampleContract.model_validate(
            {
                "schema_version": "2.0",
                "name": "alpha",
                "count": 3,
            }
        )


def test_json_schema_forbids_additional_properties() -> None:
    schema = ExampleContract.model_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"name", "count"}


def test_canonical_json_is_valid_json() -> None:
    contract = ExampleContract(name="alpha", count=3)

    decoded = json.loads(contract.canonical_json())

    assert decoded["contract_type"] == "example"
    assert decoded["schema_version"] == "1.0"
