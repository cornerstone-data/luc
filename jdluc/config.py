import dataclasses
import functools
import json
import logging
import typing

import dotenv

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Config:
    export_root: str
    harvard_dataverse_guestbook_json: str
    ingest_root: str
    number_of_dask_workers: int
    scratch_root: str
    usda_nass_api_key: str

    @classmethod
    @functools.cache
    def from_dot_env(cls) -> typing.Self:
        logger.info("Searching for a .env file")
        path_to_env = dotenv.find_dotenv(raise_error_if_not_found=True)
        logger.info(f"Loading configs from {path_to_env=:s}")
        configs = dotenv.dotenv_values(path_to_env)
        types = typing.get_type_hints(cls)
        return cls(
            **{
                field.name: types[field.name](configs[field.name.upper()])
                for field in dataclasses.fields(cls)
            }
        )

    def get_json_as_object(self, attr: str) -> dict[str, str]:
        s = getattr(self, attr)
        assert isinstance(s, str)
        return json.loads(s=s)
