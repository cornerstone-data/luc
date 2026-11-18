"""Checks on `prepare` that a real run cannot make for us.

A target standing on an anchor row that contradicts itself is the one outcome these exist to find,
so it is asserted against invented shares rather than waiting for a source to regress.
"""

import pandas

from validation import prepare, schema

# The names the findings print, for the two crops these use
COMMODITY_ID_TO_NAME = {56: "Maize (corn)", 236: "Soya beans"}


def get_deforestation(shares: dict[tuple[str, int], float]) -> pandas.DataFrame:
    """A deforestation frame carrying only what the target consistency check reads."""
    return pandas.DataFrame.from_records(
        data=[
            {
                "iso_3166": iso_3166,
                "commodity_id": commodity_id,
                "deforestation_share": share,
            }
            for (iso_3166, commodity_id), share in shares.items()
        ]
    )


def test_a_target_on_a_self_contradictory_anchor_row_is_named() -> None:
    """Keyed on the target set, because the pairs this catches are the ones no anchor reaches.

    BOL soybean and IDN maize are both real targets that no Orbae comparison covers, so a check
    hung off the comparison frame would report nothing here and read as agreement.
    """
    findings = prepare.get_target_anchor_consistency_findings(
        commodity_id_to_name=COMMODITY_ID_TO_NAME,
        deforestation=get_deforestation(
            shares={
                ("BOL", 236): 1.123,
                ("IDN", 56): 1.118,
                # Below the bound, and a real target, so it must not be reported.
                ("PRY", 236): 0.669,
                # Over the bound but not in the target set at all.
                ("XAA", 56): 9.999,
            }
        ),
    )
    assert [finding.affected_iso_3166s for finding in findings] == [("IDN",), ("BOL",)]
    assert all(finding.severity == schema.Severity.ADVISORY for finding in findings)


def test_a_target_that_arms_nothing_says_so() -> None:
    """Whether the contradictory row carries a control is the difference between the two cases."""
    (armed,) = prepare.get_target_anchor_consistency_findings(
        commodity_id_to_name=COMMODITY_ID_TO_NAME,
        deforestation=get_deforestation(shares={("PRY", 236): 1.500}),
    )
    (unarmed,) = prepare.get_target_anchor_consistency_findings(
        commodity_id_to_name=COMMODITY_ID_TO_NAME,
        deforestation=get_deforestation(shares={("IDN", 56): 1.500}),
    )
    assert "carries a control" in armed.message
    assert "arms nothing" in unarmed.message
