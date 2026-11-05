import dataclasses
import typing

import pytest

from jdluc.config import Config

# The fields a unit test needs a real value for
TEST_ENV = {"number_of_dask_workers": 2}


def get_unit_test_config() -> Config:
    """TEST_ENV over every other field left empty, so a unit test that reaches for a root fails rather than touching real storage."""
    types = typing.get_type_hints(Config)
    empty = {field.name: types[field.name]() for field in dataclasses.fields(Config)}
    return Config(**{**empty, **TEST_ENV})


@pytest.fixture(autouse=True)
def config_without_dot_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit tests need no .env; integration tests still read the real one."""
    if request.node.get_closest_marker("integration"):
        return
    config = get_unit_test_config()
    monkeypatch.setattr(Config, "from_dot_env", classmethod(lambda cls: config))
