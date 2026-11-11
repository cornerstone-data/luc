import contextlib
import dataclasses
import pathlib
import types
import typing

import dotenv
import pytest

from jdluc import config, utils

URL = "https://publisher.example/records/1/files/source.zip/content"

# The real .env parsing, past its functools.cache, captured before any test stubs it
FROM_DOT_ENV = config.Config.__dict__["from_dot_env"].__func__.__wrapped__


def get_required_lines(leave_out: str = "") -> str:
    """A .env line for every field `Config` requires, but `leave_out`."""
    types_ = typing.get_type_hints(config.Config)
    return "".join(
        f"{field.name.upper():s} = '{'2' if types_[field.name] is int else 'x'}'\n"
        for field in dataclasses.fields(config.Config)
        if field.default is dataclasses.MISSING and field.name != leave_out
    )


def read_dot_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, text: str
) -> config.Config:
    path_to_env = tmp_path / ".env"
    path_to_env.write_text(text)
    monkeypatch.setattr(dotenv, "find_dotenv", lambda **kwargs: str(path_to_env))
    return FROM_DOT_ENV(config.Config)


@pytest.fixture
def requested_urls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the publisher: every GET serves b"source"; return the URLs it was asked for."""
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


def set_source_cache_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, root: str
) -> None:
    configured = dataclasses.replace(
        read_dot_env(
            monkeypatch=monkeypatch, tmp_path=tmp_path, text=get_required_lines()
        ),
        source_cache_root=root,
    )
    monkeypatch.setattr(
        config.Config, "from_dot_env", classmethod(lambda cls: configured)
    )


def test_a_miss_downloads_and_stores_and_a_hit_skips_the_publisher(
    monkeypatch: pytest.MonkeyPatch,
    requested_urls: list[str],
    tmp_path: pathlib.Path,
) -> None:
    set_source_cache_root(
        monkeypatch=monkeypatch, root=str(tmp_path / "cache"), tmp_path=tmp_path
    )
    for name in ("first", "second"):
        utils.save_remote_url_to_local_path(
            cache_key="zenodo/1/source.zip",
            local_path=str(tmp_path / name),
            params={},
            remote_url=URL,
        )
        assert (tmp_path / name).read_bytes() == b"source"
    assert requested_urls == [URL]
    assert (tmp_path / "cache/zenodo/1/source.zip").read_bytes() == b"source"


@pytest.mark.parametrize(
    ("root", "cache_key"),
    [
        pytest.param("", "zenodo/1/source.zip", id="root-unset"),
        pytest.param("cache", None, id="no-key"),
    ],
)
def test_without_a_root_or_a_key_every_call_downloads(
    monkeypatch: pytest.MonkeyPatch,
    requested_urls: list[str],
    tmp_path: pathlib.Path,
    root: str,
    cache_key: str | None,
) -> None:
    set_source_cache_root(
        monkeypatch=monkeypatch,
        root=str(tmp_path / root) if root else "",
        tmp_path=tmp_path,
    )
    for name in ("first", "second"):
        utils.save_remote_url_to_local_path(
            cache_key=cache_key,
            local_path=str(tmp_path / name),
            params={},
            remote_url=URL,
        )
    assert requested_urls == [URL, URL]
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    ("extra_line", "expected"),
    [
        pytest.param("", "", id="absent"),
        pytest.param("SOURCE_CACHE_ROOT = '/sources'\n", "/sources", id="set"),
    ],
)
def test_source_cache_root_may_be_left_out_of_dot_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    extra_line: str,
    expected: str,
) -> None:
    configured = read_dot_env(
        monkeypatch=monkeypatch,
        tmp_path=tmp_path,
        text=get_required_lines() + extra_line,
    )
    assert configured.source_cache_root == expected


def test_a_required_key_may_not_be_left_out_of_dot_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    with pytest.raises(KeyError, match="SCRATCH_ROOT"):
        read_dot_env(
            monkeypatch=monkeypatch,
            tmp_path=tmp_path,
            text=get_required_lines(leave_out="scratch_root"),
        )
