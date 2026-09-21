import collections.abc
import concurrent.futures
import itertools
import re
import time

import pytest

from jdluc.utils import (
    iter_sharded,
    threadsafe_cache,
)


def test_threadsafe_cache_serial() -> None:
    @threadsafe_cache
    def now() -> float:
        time.sleep(0.2)
        return time.monotonic()

    result = now()
    assert now() == result
    assert now() == result
    assert now() == result
    assert now() == result


def test_threadsafe_cache_parallel() -> None:
    @threadsafe_cache
    def now() -> float:
        time.sleep(0.2)
        return time.monotonic()

    results: set[float] = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures: list[concurrent.futures.Future[float]] = []
        for _ in range(25):
            futures.append(executor.submit(now))
        for future in concurrent.futures.as_completed(futures):
            results.add(future.result())
    assert len(results) == 1


@pytest.mark.parametrize(
    ("modulus", "residues", "values", "expected"),
    (
        (1, None, range(5), range(5)),
        (1, (0,), range(5), range(5)),
        (2, (0,), range(5), (0, 2, 4)),
        (2, (1,), range(5), (1, 3)),
        (2, (0, 1), range(5), range(5)),
        (3, (0,), range(5), (0, 3)),
        (3, (1,), range(5), (1, 4)),
        (3, (2,), range(5), (2,)),
    ),
)
def test_iter_sharded(
    modulus: int,
    residues: collections.abc.Sequence[int] | None,
    values: collections.abc.Sequence[int],
    expected: collections.abc.Sequence[int],
) -> None:
    result = iter_sharded(
        modulus=modulus,
        residues=residues,
        values=values,
    )
    assert list(result) == list(expected)


def test_iter_sharded_is_disjoint() -> None:
    shards = [
        list(iter_sharded(modulus=3, residues=(residue,), values=range(100)))
        for residue in range(3)
    ]
    for left, right in itertools.combinations(shards, r=2):
        assert set(left).isdisjoint(right)
    assert {value for shard in shards for value in shard} == set(range(100))


def test_iter_sharded_raises() -> None:
    with pytest.raises(
        AssertionError, match=re.escape("residues=(2,) exceed modulus=2")
    ):
        list(iter_sharded(modulus=2, residues=(2,), values=()))
