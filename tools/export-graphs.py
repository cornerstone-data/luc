import collections.abc
import csv
import dataclasses
import io
import logging
import tempfile
import typing
import uuid
import zipfile

import numpy
import pandas

from jdluc import attribute, statistical, trace
from jdluc.datasets import ifpri_mapspam, worldbank_jurisdictions

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Record:
    admin_id: str
    crop_name: str
    emissions_factor_kgco2e_per_kg: float
    emissions_mt: float
    methodology: str
    production_kg: float

    @classmethod
    def from_dataframe(cls, df: pandas.DataFrame) -> typing.Self:
        ((_, srs),) = df.iterrows()
        return cls(
            admin_id=str(srs.admin_id),
            crop_name=str(srs.crop_name),
            emissions_factor_kgco2e_per_kg=float(srs.emissions_factor_kgco2e_per_kg),
            emissions_mt=numpy.nan_to_num(srs.emissions_mt),
            methodology=str(srs.methodology),
            production_kg=float(srs.production_kg),
        )

    @property
    def crop_description(self) -> str:
        crop_name_2005 = statistical.Crop[self.crop_name].value
        return ifpri_mapspam.Crop2005[crop_name_2005].value

    @property
    def payload(self) -> dict[str, float | str]:
        emissions_kgco2e = self.emissions_factor_kgco2e_per_kg * self.production_kg
        return {
            "identifier": str(uuid.uuid4()),
            "name": "Land use change",
            "node_type": "processing",
            "lifecycle_stage": "A3",
            "downstream_node_identifier": "",
            "output_material_name": "Land use change",
            "output_amount": self.production_kg,
            "output_amount_unit": "kg",
            "emissions_kgco2e": emissions_kgco2e,
            "cumulative_flag_emissions_kgco2e": emissions_kgco2e,
            "land_use_change_ghg_emissions_kgco2e": 0,
            "cumulative_land_use_change_ghg_emissions_kgco2e": emissions_kgco2e,
            "land_management_fossil_ghg_emissions_kgco2e": 0,
            "cumulative_land_management_fossil_ghg_emissions_kgco2e": "",
            "land_management_excl_biogenic_co2_kgco2e": "",
            "cumulative_land_management_excl_biogenic_co2_kgco2e": "",
            "flag_is_cumulative": False,
            "is_flag_node": False,
            "conversion_factor_value": "",
            "conversion_factor_unit": 1,
        }

    @property
    def csv(self) -> str:
        buffer = io.StringIO()
        w = csv.DictWriter(buffer, fieldnames=list(self.payload))
        w.writeheader()
        w.writerow(self.payload)
        return buffer.getvalue()


PROVINCE_NAME_BLOCKLIST = {
    "Brazil | Name Unknown",
}

METHODOLOGY_TO_ISO_3166_TO_CROP_NAMES = {
    attribute.Methodology.JURISDICTIONAL_DIRECT: {"USA": ("MAIZE", "SOYBEAN", "WHEAT")},
    attribute.Methodology.STATISTICAL: {
        "BRA": ("MAIZE", "SOYBEAN", "OILPALM"),
        "BOL": ("SOYBEAN", "PLANTAIN", "BANANA"),
        "ARG": ("SOYBEAN",),
        "CAN": ("SOYBEAN",),
        "IDN": ("OILPALM",),
        "PNG": ("OILPALM",),
        "GHA": ("RICE", "WHEAT"),
        "VNM": ("RICE", "SORGHUM"),
        "MYS": ("RICE", "SORGHUM", "OILPALM"),
        "CUB": ("SUGARCANE", "RICE", "BANANA"),
        "ZAF": ("RICE", "WHEAT", "SORGHUM"),
    },
}


def subset_to_crop_names(
    df: pandas.DataFrame, crop_names: tuple[str, ...]
) -> pandas.DataFrame:
    ret = df[df.index.get_level_values("crop_name").isin(crop_names)]
    missing = sorted(
        set(crop_names) - set(map(str, ret.index.get_level_values("crop_name")))
    )
    if missing:
        logger.warning(f"Workflow returned no rows for {missing=}")
    return ret


def iter_dfs(
    iso_3166_to_crop_names: dict[str, tuple[str, ...]],
    methodology: attribute.Methodology,
) -> collections.abc.Iterator[pandas.DataFrame]:
    # The full crop list, so this reuses the caches an ordinary trace.py run populates
    workflow_crop_names = attribute.get_crop_names(methodology=methodology)
    for iso_3166, crop_names in iso_3166_to_crop_names.items():
        logger.info(
            f"Tracing {methodology.name:s}/{iso_3166:s} with "
            f"{len(workflow_crop_names):d} crops, tabulating {len(crop_names):d}"
        )
        yield subset_to_crop_names(
            df=trace.workflow(
                crop_names=workflow_crop_names,
                iso_3166s=(iso_3166,),
                methodology=methodology,
                skip_glad_crop_filter=False,
            ),
            crop_names=crop_names,
        )


def iter_path_record(
    df: pandas.DataFrame,
) -> collections.abc.Generator[tuple[tuple[str, ...], Record]]:
    for crop_name, crop_df in df.reset_index(drop=False).groupby(by="crop_name"):
        nationals = crop_df[
            crop_df.admin_level == worldbank_jurisdictions.AdminLevel.NATIONAL.name
        ]
        for national_name, national_df in nationals.groupby(by="jurisdiction_name"):
            record = Record.from_dataframe(df=national_df)
            yield (record.crop_description, national_name), record
            provincials = crop_df[
                (
                    crop_df.admin_level
                    == worldbank_jurisdictions.AdminLevel.PROVINCIAL.name
                )
                & (crop_df.jurisdiction_name.str.startswith(national_name))
            ]
            for provincial_name, provincial_df in provincials.groupby(
                "jurisdiction_name"
            ):
                if provincial_name in PROVINCE_NAME_BLOCKLIST:
                    logger.warning(
                        f"Skipping {crop_name=:s}/{provincial_name=:s} because it is blocklisted"
                    )
                    continue
                else:
                    record = Record.from_dataframe(df=provincial_df)
                    yield (
                        (record.crop_description, *provincial_name.split(" | ")),
                        record,
                    )


def write_data_to_graph(
    dfs: list[pandas.DataFrame],
) -> None:
    logger.info("Populating graph zipfile with records")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for df in dfs:
            for path_tokens, record in iter_path_record(df=df):
                arcname = "/".join((*path_tokens, record.crop_description + ".csv"))
                zf.writestr(arcname, record.csv)

    with tempfile.NamedTemporaryFile(delete=False, mode="wb", suffix=".zip") as fp:
        logger.info(f"Writing graph to {fp.name=:s}")
        fp.write(buffer.getvalue())


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    write_data_to_graph(
        dfs=[
            pandas.concat(
                iter_dfs(
                    iso_3166_to_crop_names=iso_3166_to_crop_names,
                    methodology=methodology,
                )
            )
            for methodology, iso_3166_to_crop_names in METHODOLOGY_TO_ISO_3166_TO_CROP_NAMES.items()
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
