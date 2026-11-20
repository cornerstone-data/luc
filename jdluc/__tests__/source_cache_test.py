import contextlib
import dataclasses
import pathlib
import types

import pytest

from jdluc import utils
from jdluc.__tests__.storage_test import get_unit_test_config
from jdluc.config import Config

URL = "https://publisher.example/records/1/files/lon_%2B000.zip/content"


@pytest.fixture
def requested_urls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> list[str]:
    """SCRATCH_ROOT under tmp_path, and a stubbed publisher; return the URLs it was asked for."""
    config = dataclasses.replace(
        get_unit_test_config(), scratch_root=str(tmp_path / "scratch")
    )
    monkeypatch.setattr(Config, "from_dot_env", classmethod(lambda cls: config))
    urls: list[str] = []

    @contextlib.contextmanager
    def request(url: str, **kwargs: object):  # type: ignore[no-untyped-def]
        urls.append(url)
        yield types.SimpleNamespace(
            iter_content=lambda chunk_size: iter([b"source"]),
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        utils, "get_requests_session", lambda: types.SimpleNamespace(request=request)
    )
    return urls


def download(tmp_path: pathlib.Path, name: str, immutable: bool) -> bytes:
    local_path = tmp_path / name
    utils.save_remote_url_to_local_path(
        immutable=immutable, local_path=str(local_path), params={}, remote_url=URL
    )
    return local_path.read_bytes()


def test_immutable_download_is_fetched_once_then_shared(
    requested_urls: list[str], tmp_path: pathlib.Path
) -> None:
    assert download(tmp_path, "first", immutable=True) == b"source"
    assert (
        tmp_path
        / "scratch/sources/publisher.example/records/1/files/lon_+000.zip/content"
    ).read_bytes() == b"source"
    assert download(tmp_path, "second", immutable=True) == b"source"
    assert requested_urls == [URL]


def test_other_downloads_are_fetched_every_time(
    requested_urls: list[str], tmp_path: pathlib.Path
) -> None:
    download(tmp_path, "first", immutable=False)
    download(tmp_path, "second", immutable=False)
    assert requested_urls == [URL, URL]
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize(
    "params, remote_url",
    [({"a": "1"}, URL), ({}, f"{URL:s}?download=1"), ("a=1", URL)],
)
def test_immutable_download_takes_no_parameters(
    requested_urls: list[str],
    tmp_path: pathlib.Path,
    params: dict[str, str] | str,
    remote_url: str,
) -> None:
    with pytest.raises(ValueError, match="takes no parameters"):
        utils.save_remote_url_to_local_path(
            immutable=True,
            local_path=str(tmp_path / "source"),
            params=params,
            remote_url=remote_url,
        )
    assert requested_urls == []
