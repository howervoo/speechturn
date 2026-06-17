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
