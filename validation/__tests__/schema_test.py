"""Pin schema.py's copied constants to the sources they copy.

`validation/schema.py` declares the admin-level names, the methodology names and the
tonne-to-kilogram factor itself rather than importing them, so the reporting path needs only pandas:
importing `worldbank_jurisdictions` for an enum member would pull geopandas onto it. A test may
import jdluc freely, not being that path, so these assert the copies still agree.

The stored strings are the ones that matter. A renamed `AdminLevel` or `Methodology` member would
leave `prepare`'s filters matching nothing, reporting an empty table as agreement, which nothing
downstream would catch. `KG_PER_TONNE` cannot realistically drift and is pinned because it is free.

`trace.ADDITIVE_COLUMNS` is named there too and so is pinnable, but nothing here copies it yet.
`prepare` will, for the national totals, and it should be pinned at the same time: a column added to
the sum in trace and not here would simply not be summed, with nothing to say so.

tools/measure-drift.py keeps its own copies too. They are not pinned here: its filename has a hyphen
so it cannot be imported, and policing a sibling tool's internals is not this suite's job. Both
tools copy from the same source, so pinning each to jdluc keeps them consistent with each other.
"""

from jdluc import attribute, trace
from jdluc.datasets import worldbank_jurisdictions
from validation import schema


def test_admin_level_names_match_the_enum() -> None:
    """The table stores AdminLevel's member names, and prepare filters on these strings."""
    assert worldbank_jurisdictions.AdminLevel.NATIONAL.name == schema.NATIONAL
    assert worldbank_jurisdictions.AdminLevel.PROVINCIAL.name == schema.PROVINCIAL


def test_methodology_names_match_the_enum() -> None:
    """Both legs share one capture, so these strings are what separates them."""
    assert attribute.Methodology.STATISTICAL.name == schema.STATISTICAL
    assert (
        attribute.Methodology.JURISDICTIONAL_DIRECT.name == schema.JURISDICTIONAL_DIRECT
    )


def test_every_methodology_is_named_here() -> None:
    """A third leg would otherwise be silently absent from every comparison."""
    assert {schema.STATISTICAL, schema.JURISDICTIONAL_DIRECT} == {
        methodology.name for methodology in attribute.Methodology
    }


def test_kg_per_tonne_matches_trace() -> None:
    """trace applies this to produce production_kg; schema applies it to check the factor."""
    assert schema.KG_PER_TONNE == trace.KG_PER_TONNE


def test_canonical_key_matches_the_index_trace_writes() -> None:
    """Re-keying onto the wrong names would drop every row of a comparison."""
    assert schema.CANONICAL_KEY == trace.CANONICAL_KEY
