import pytest

from chai_mlx.config import ChaiConfig


@pytest.fixture
def cfg() -> ChaiConfig:
    return ChaiConfig()
