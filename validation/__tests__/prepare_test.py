"""Checks on the yield cross-check that a real run cannot make for us.

The live run currently finds every crop in agreement, which is the answer we want but leaves the
failure paths unexercised: nothing in it proves a product-form mismatch would actually be caught. A
mismatch is the one outcome this check exists to find, so it is asserted here against invented
ratios rather than waiting for a source to regress.
"""

import pandas
import pytest

from validation import prepare, schema


def get_comparison(ratios: dict[str, list[float]]) -> pandas.DataFrame:
    """A comparison frame carrying only what `iter_yield_agreements` reads."""
    return pandas.DataFrame.from_records(
        data=[
            {"crop_name": crop_name, "ratio": ratio}
            for crop_name, crop_ratios in ratios.items()
            for ratio in crop_ratios
        ]
    )


def test_a_crop_short_of_the_minimum_is_not_medianed() -> None:
    """Two countries agreeing says nothing about a crop, and would read as though it did."""
    agreements = tuple(
        prepare.iter_yield_agreements(
            comparison=get_comparison(
                ratios={
                    "XAA": [1.0] * (prepare.MINIMUM_COUNTRIES - 1),
                    "XAB": [1.0] * prepare.MINIMUM_COUNTRIES,
                }
            )
        )
    )
    assert [agreement.crop_name for agreement in agreements] == ["XAB"]


def test_an_outlier_country_does_not_move_the_median() -> None:
    """One near-zero denominator produces a ratio in the hundreds; a mean would follow it."""
    (agreement,) = prepare.iter_yield_agreements(
        comparison=get_comparison(ratios={"XAA": [1.0, 1.0, 1.0, 1.0, 1.0, 400.0]})
    )
    assert agreement.median_ratio == 1.0
    assert agreement.highest_ratio == 400.0


@pytest.mark.parametrize(
    ("ratio", "is_mismatch", "severity"),
    (
        pytest.param(1.00, False, None, id="parity-says-nothing-and-must-say-nothing"),
        pytest.param(0.93, False, None, id="mapspam-runs-7-percent-under-faostat"),
        pytest.param(0.86, False, None, id="just-inside-the-tolerance"),
        pytest.param(
            0.80,
            False,
            schema.Severity.ADVISORY,
            id="a-denominator-difference-is-the-same-quantity-differently-estimated",
        ),
        pytest.param(
            0.50,
            False,
            schema.Severity.ADVISORY,
            id="the-product-form-bound-is-inclusive",
        ),
        pytest.param(2.00, False, schema.Severity.ADVISORY, id="at-the-top-as-well"),
        pytest.param(
            2.01,
            True,
            schema.Severity.BLOCKING,
            id="just-past-it-is-a-different-quantity",
        ),
        pytest.param(
            0.20,
            True,
            schema.Severity.BLOCKING,
            id="palm-oil-against-fruit-bunches-is-a-milling-yield",
        ),
    ),
)
def test_a_yield_disagreement_is_graded_by_how_far_it_is(
    ratio: float, is_mismatch: bool, severity: schema.Severity | None
) -> None:
    """A mismatch is BLOCKING and a disagreement ADVISORY: the first was never like-for-like.

    0.86 rather than the 0.85 the tolerance would suggest: 1 - 0.85 is 0.15000000000000002 in
    IEEE 754, so the exact bound fires, and pinning that as inclusive would encode a float artefact
    as intent.
    """
    (agreement,) = prepare.iter_yield_agreements(
        comparison=get_comparison(ratios={"XAA": [ratio] * prepare.MINIMUM_COUNTRIES})
    )
    assert agreement.is_product_form_mismatch == is_mismatch
    findings = prepare.get_yield_findings(agreements=(agreement,), year=2020)
    if severity is None:
        assert findings == []
    else:
        (finding,) = findings
        assert finding.severity == severity


def test_every_unpaired_crop_gets_a_reason() -> None:
    """An unexplained crop would be dropped from the report, reading as one that agreed."""
    wri_yields = pandas.DataFrame.from_records(
        data=[
            {"crop_name": name}
            for name in ("MAIZ", "OOIL", "RCOF", "CITR", "SESA", "BARL")
        ]
    )
    unpaired = prepare.get_unpaired_crop_names(
        comparison=get_comparison(ratios={"MAIZ": [1.0], "SESA": [1.0]}),
        wri_yields=wri_yields,
    )
    assert unpaired == {
        schema.UnpairedReason.SPAM_GROUP: ("OOIL",),
        schema.UnpairedReason.SPAM_SPLIT: ("RCOF",),
        schema.UnpairedReason.TOO_FEW_COUNTRIES: ("BARL",),
        schema.UnpairedReason.UNMAPPED: ("CITR",),
    }


def get_deforestation(shares: dict[tuple[str, str], float]) -> pandas.DataFrame:
    """A deforestation frame carrying only what the target consistency check reads."""
    return pandas.DataFrame.from_records(
        data=[
            {"iso_3166": iso_3166, "crop_name": crop_name, "deforestation_share": share}
            for (iso_3166, crop_name), share in shares.items()
        ]
    )


def test_a_target_on_a_self_contradictory_anchor_row_is_named() -> None:
    """Keyed on the target set, because the pairs this catches are the ones no anchor reaches.

    BOL SOYBEAN and IDN MAIZE are both real targets that no Orbae comparison covers, so a check
    hung off the comparison frame would report nothing here and read as agreement. BOL is the pair
    whose crop code differs from its name (SOYB), which is where a keying mistake would show.
    """
    findings = prepare.get_target_anchor_consistency_findings(
        deforestation=get_deforestation(
            shares={
                ("BOL", "SOYB"): 1.123,
                ("IDN", "MAIZ"): 1.118,
                # Below the bound, and a real target, so it must not be reported.
                ("PRY", "SOYB"): 0.669,
                # Over the bound but not in the target set at all.
                ("XAA", "MAIZ"): 9.999,
            }
        )
    )
    assert [finding.affected_iso_3166s for finding in findings] == [("IDN",), ("BOL",)]
    assert all(finding.severity == schema.Severity.ADVISORY for finding in findings)


def test_a_target_that_arms_nothing_says_so() -> None:
    """Whether the contradictory row carries a control is the difference between the two cases."""
    (armed,) = prepare.get_target_anchor_consistency_findings(
        deforestation=get_deforestation(shares={("PRY", "SOYB"): 1.500})
    )
    (unarmed,) = prepare.get_target_anchor_consistency_findings(
        deforestation=get_deforestation(shares={("IDN", "MAIZ"): 1.500})
    )
    assert "carries a control" in armed.message
    assert "arms nothing" in unarmed.message
