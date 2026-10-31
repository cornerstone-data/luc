import pytest

from jdluc.datasets import NAME_TO_CLS, base


@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(dataset, id=name)
        for name, dataset in NAME_TO_CLS.items()
        if isinstance(dataset, base.TabularDataset)
    ],
)
def test_a_tabular_datasets_are_hashable(dataset: base.TabularDataset) -> None:
    assert hash(dataset)
