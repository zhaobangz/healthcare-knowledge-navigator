from __future__ import annotations

import pytest

from hkn.ingest.parse_html import HtmlParserConfig, ParseError, parse_html
from hkn.ingest.parse_jats import LicenseRejected, classify_license, parse_jats
from hkn.ingest.parse_spl import parse_spl

MMWR_LIKE = """
<html><body><nav><h2>Site navigation</h2><p>Home</p></nav>
<main>
  <h1>STI Treatment Guidelines</h1>
  <h2>Summary</h2><p>Not part of the indexed text.</p>
  <h2>Introduction</h2>
  <p>These guidelines update prior recommendations (<i>1</i>,<i>2</i>).</p>
  <h2>Gonococcal Infections</h2>
  <h3 id="gc-uncomplicated">Uncomplicated Gonococcal Infection</h3>
  <table><caption>Recommended Regimen</caption><tbody>
    <tr><td><b>Recommended Regimen</b></td></tr>
    <tr><td><b>Ceftriaxone</b> 500 mg IM in a single dose</td></tr>
  </tbody></table>
  <table><caption>Alternative Regimens</caption><tbody>
    <tr><td>Gentamicin 240 mg IM in a single dose</td></tr>
    <tr><td><i>plus</i></td></tr>
    <tr><td>Azithromycin 2 g orally in a single dose</td></tr>
  </tbody></table>
  <p>MIC = 0.03 <i>µ</i>g/mL<span class="sr-only">start highlight</span>.</p>
  <h3>Work Group Members</h3><p>Dr. Example</p>
  <h2>References</h2><p>1. Some citation.</p>
</main></body></html>
"""

CONFIG = HtmlParserConfig(
    root="main",
    start_heading="^(Introduction|Background)$",
    stop_headings=["^References$"],
    skip_headings=["Work ?Group"],
    strip_numeric_italics=True,
)


def test_html_parser_respects_start_stop_skip_and_strips_reference_markers() -> None:
    doc = parse_html(MMWR_LIKE, CONFIG)
    headings = [s.heading_path for s in doc.sections]
    assert headings == [
        ["Introduction"],
        ["Gonococcal Infections", "Uncomplicated Gonococcal Infection"],
    ]
    intro = doc.sections[0].blocks[0].text
    assert intro == "These guidelines update prior recommendations."
    all_text = " ".join(b.text for s in doc.sections for b in s.blocks)
    assert "Not part of the indexed text" not in all_text
    assert "Dr. Example" not in all_text and "Some citation" not in all_text
    assert "start highlight" not in all_text


def test_html_tables_become_self_describing_lines_and_connectors_merge() -> None:
    section = parse_html(MMWR_LIKE, CONFIG).sections[1]
    assert section.anchor == "gc-uncomplicated"
    kinds_texts = [(b.kind, b.text) for b in section.blocks]
    assert ("caption", "Recommended Regimen") in kinds_texts
    assert ("table_row", "Ceftriaxone 500 mg IM in a single dose") in kinds_texts
    assert (
        "table_row",
        "Gentamicin 240 mg IM in a single dose plus Azithromycin 2 g orally in a single dose",
    ) in kinds_texts
    # Inline tags must not split words: "µg/mL", not "µ g/mL".
    assert any("0.03 µg/mL." in t for _, t in kinds_texts)


def test_multi_column_table_rows_carry_their_headers() -> None:
    html = """<main><h2>Background</h2><table>
      <thead><tr><th>Population</th><th>Recommendation</th><th>Grade</th></tr></thead>
      <tbody><tr><td>Adults 45-49</td><td>Screen for CRC</td><td>B</td></tr></tbody>
    </table></main>"""
    doc = parse_html(html, HtmlParserConfig(start_heading="^Background$"))
    assert (
        doc.sections[0].blocks[0].text
        == "Population: Adults 45-49; Recommendation: Screen for CRC; Grade: B"
    )


def test_missing_start_heading_fails_loudly() -> None:
    with pytest.raises(ParseError):
        parse_html(
            "<main><h2>Other</h2><p>x</p></main>", HtmlParserConfig(start_heading="^Introduction$")
        )


SPL = """<?xml version="1.0"?>
<document xmlns="urn:hl7-org:v3"><title>Ceftriaxone Injection</title>
<component><structuredBody>
  <component><section ID="s1"><code code="34067-9"/><title>INDICATIONS AND USAGE</title>
    <text><paragraph>Indicated for gonorrhea.</paragraph></text>
    <component><section><code code="42229-5"/><title>Adults</title>
      <text><list><item>1 to 2 grams daily.</item></list></text></section></component>
  </section></component>
  <component><section><code code="51945-4"/><title>PACKAGE LABEL</title>
    <text><paragraph>NDC 1234</paragraph></text></section></component>
</structuredBody></component></document>"""


def test_spl_parser_keeps_unclassified_subsections_and_skips_package_label() -> None:
    doc = parse_spl(SPL)
    assert [s.heading_path for s in doc.sections] == [
        ["INDICATIONS AND USAGE"],
        ["INDICATIONS AND USAGE", "Adults"],
    ]
    assert doc.sections[1].blocks[0].kind == "list_item"
    assert all("NDC" not in b.text for s in doc.sections for b in s.blocks)


def jats(license_xml: str) -> str:
    return f"""<?xml version="1.0"?>
<article xmlns:xlink="http://www.w3.org/1999/xlink" article-type="research-article">
<front><article-meta>
  <title-group><article-title>A randomized controlled trial of doxycycline PEP</article-title></title-group>
  <contrib-group><contrib contrib-type="author"><name><surname>Lee</surname><given-names>A</given-names></name></contrib></contrib-group>
  <pub-date><year>2023</year></pub-date>
  <permissions>{license_xml}</permissions>
  <abstract><p>Doxycycline reduced incident STIs <xref ref-type="bibr" rid="r1">[1]</xref>.</p></abstract>
</article-meta></front>
<body><sec id="s1"><title>Methods</title><p>We randomized 501 participants.</p></sec></body>
<back><ref-list><ref id="r1">Citation</ref></ref-list></back>
</article>"""


def test_jats_parser_admits_cc_by_and_extracts_metadata() -> None:
    doc = parse_jats(
        jats(
            '<license xlink:href="https://creativecommons.org/licenses/by/4.0/"><license-p>CC BY</license-p></license>'
        )
    )
    assert doc.doc_type == "rct"
    assert doc.pub_year == 2023
    assert doc.authors == "Lee A"
    assert [s.heading_path for s in doc.sections] == [["Abstract"], ["Methods"]]
    assert doc.sections[0].blocks[0].text == "Doxycycline reduced incident STIs."


def test_jats_parser_rejects_non_commercial_license() -> None:
    with pytest.raises(LicenseRejected):
        parse_jats(jats('<license xlink:href="https://creativecommons.org/licenses/by-nc/4.0/"/>'))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://creativecommons.org/licenses/by/4.0/", "cc-by"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", None),
        ("This is an open access article under the Creative Commons Attribution License", "cc-by"),
        ("Creative Commons Attribution-NonCommercial License", None),
        ("© 2023 Publisher. All rights reserved.", None),
    ],
)
def test_classify_license(text: str, expected: str | None) -> None:
    assert classify_license(text) == expected
