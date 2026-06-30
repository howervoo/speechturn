import json
from pathlib import Path

import pytest

from tests.api_contract import module_contract

PACKAGE = Path(__file__).parents[1] / "src"
CONTRACTS = json.loads((Path(__file__).parent / "fixtures/api-contracts.json").read_text())


@pytest.mark.parametrize("name", sorted(CONTRACTS))
def test_public_api_contract(name):
    record = CONTRACTS[name]
    assert module_contract(PACKAGE / record["path"]) == record["contract"]


def test_every_module_has_a_golden_contract():
    paths = {p.relative_to(PACKAGE).as_posix() for p in (PACKAGE / "speechturn").rglob("*.py")}
    assert {record["path"] for record in CONTRACTS.values()} == paths


def test_every_serialized_field_has_invalid_input_coverage():
    from dataclasses import fields

    from tests.test_input_contracts import CASES, CONSTRUCTORS

    expected = {(name, field.name) for name, cls in CONSTRUCTORS.items() for field in fields(cls)}
    assert {(case["class"], case["field"]) for case in CASES} == expected
    assert len({case["id"] for case in CASES}) == len(CASES)
