from jdluc.datasets.glad_glcluc import flatten_ranges


def test_flatten_ranges() -> None:
    assert flatten_ranges(range(5), range(5, 10), range(10, 15)) == list(range(15))
