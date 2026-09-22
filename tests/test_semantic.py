"""Prose extraction, retrieval and contradiction candidates.

The prose tests use the shapes a real page actually produced. A text extraction
proxy returns the whole page, so what comes back is an article wrapped in a
navigation menu, a consent banner and a legal notice, and every one of those
defeats an obvious filter in its own way: the menu is long, the banner is
written in full sentences, and the disclaimer is both.

Getting this wrong is not cosmetic. A run stored 2,900 words of navigation from
heart.org as a citable source, reported five sources gathered, and would have
let an agent cite a cookie notice.
"""
from __future__ import annotations

from mas.kernel.semantic import (
    MIN_ARTICLE_CHARS,
    article_text,
    best_passages,
    candidate_pairs,
    rank_evidence,
    split_paragraphs,
)

ARTICLE = (
    "Drinking three to five cups of coffee a day is associated with a lower risk of "
    "cardiovascular disease in large observational studies. The effect size is modest "
    "and confounding by lifestyle has not been ruled out."
)
SECOND = (
    "Unfiltered coffee raises LDL cholesterol because it retains cafestol and kahweol, "
    "the diterpenes that a paper filter removes. The difference is measurable within "
    "four weeks of switching brewing method."
)

# A menu, as markdown links. Long, many words, and not a sentence anywhere.
NAV = (
    "[Health Topics](https://x.test/a)\n[Aortic Aneurysm](https://x.test/b)\n"
    "[Arrhythmia](https://x.test/c)\n[Atrial Fibrillation](https://x.test/d)\n"
    "[Cardiac Arrest](https://x.test/e)\n[Cardiac Rehab](https://x.test/f)\n"
    "[Cardiomyopathy](https://x.test/g)\n[Cholesterol](https://x.test/h)"
)
# A consent banner. Full sentences, plenty long, and still not content.
CONSENT = (
    "We process your personal information to measure and improve our sites and "
    "service, to assist our marketing campaigns and to provide personalised content. "
    "As a California consumer you have the right to opt-out at any time."
)
DISCLAIMER = (
    "This link is provided for convenience only and is not an endorsement of either "
    "the linked-to entity or any product or service, and all medical information on "
    "this website has been reviewed and approved by the association."
)


def page(*blocks: str) -> str:
    return "\n\n".join(blocks)


def test_an_article_paragraph_survives():
    assert split_paragraphs(page(ARTICLE)) == [ARTICLE]


def test_a_navigation_menu_is_dropped():
    """It is long and wordy. What it has not got is a terminated sentence."""
    assert split_paragraphs(page(NAV)) == []


def test_a_consent_banner_is_dropped_despite_being_sentences():
    assert split_paragraphs(page(CONSENT)) == []


def test_a_legal_disclaimer_is_dropped():
    assert split_paragraphs(page(DISCLAIMER)) == []


def test_the_article_is_recovered_from_a_page_full_of_furniture():
    body = page(NAV, CONSENT, ARTICLE, DISCLAIMER, SECOND, NAV)
    kept = split_paragraphs(body)
    assert kept == [ARTICLE, SECOND]


def test_markdown_link_targets_are_stripped_from_kept_prose():
    linked = (
        "Coffee intake was measured with a [food frequency questionnaire]"
        "(https://x.test/ffq) in the cohort, and the association held after adjustment "
        "for smoking and physical activity across every subgroup examined."
    )
    kept = split_paragraphs(page(linked))
    assert len(kept) == 1
    assert "https://x.test/ffq" not in kept[0]
    assert "food frequency questionnaire" in kept[0]


def test_a_page_that_is_only_furniture_yields_too_little_to_store():
    """The exact case that stored a navigation bar as a source."""
    assert len(article_text(page(NAV, CONSENT, DISCLAIMER, NAV))) < MIN_ARTICLE_CHARS


THIRD = (
    "Decaffeinated coffee shows the same association in most cohorts, which argues "
    "that the effect is not attributable to caffeine alone. Polyphenol content is the "
    "usual explanation offered, though no trial has isolated it."
)


def test_a_page_with_a_real_article_yields_enough_to_store():
    body = page(NAV, ARTICLE, SECOND, THIRD, ARTICLE.replace("coffee", "tea"), CONSENT)
    assert len(article_text(body)) >= MIN_ARTICLE_CHARS


def test_best_passages_returns_the_relevant_paragraph():
    body = page(ARTICLE, SECOND, "Unrelated filler about municipal parking policy that "
                "goes on for long enough to pass the length filter and contains a full "
                "sentence so that it is not dropped as furniture.")
    passages = best_passages(body, "does unfiltered coffee affect cholesterol", limit=1)
    assert len(passages) == 1
    assert "cholesterol" in passages[0].lower()


def test_best_passages_survives_a_body_with_no_paragraphs():
    assert best_passages("", "anything") == []
    assert best_passages("short", "anything") == ["short"]


def test_rank_evidence_orders_by_subject_without_stored_vectors():
    """No embeddings stored means token overlap, which must still work."""
    rows = [
        {"ref": "S1", "title": "Municipal parking policy", "snippet": "parking permits and zones"},
        {"ref": "S2", "title": "Coffee and cholesterol", "snippet": "unfiltered coffee raises LDL"},
    ]
    ranked = rank_evidence(rows, "coffee cholesterol", limit=2)
    assert ranked[0][0]["ref"] == "S2"


def test_candidate_pairs_finds_claims_about_the_same_thing():
    claims = [
        {"id": "C1", "text": "The central bank cut the policy rate in June.", "sources": ["S1"]},
        {"id": "C2", "text": "The central bank held the policy rate in June.", "sources": ["S2"]},
        {"id": "C3", "text": "Rainfall in the northeast was above average.", "sources": ["S3"]},
    ]
    pairs = candidate_pairs(claims, threshold=0.5)
    found = {(a["id"], b["id"]) for a, b, _ in pairs}
    assert ("C1", "C2") in found
    # The unrelated claim is not paired with either.
    assert not any("C3" in pair for pair in found)


def test_candidate_pairs_skips_claims_from_the_same_source():
    """Two claims from one page disagreeing is a reading error, not a conflict."""
    claims = [
        {"id": "C1", "text": "The central bank cut the policy rate in June.", "sources": ["S1"]},
        {"id": "C2", "text": "The central bank held the policy rate in June.", "sources": ["S1"]},
    ]
    assert candidate_pairs(claims, threshold=0.1) == []


def test_candidate_pairs_needs_two_claims():
    assert candidate_pairs([]) == []
    assert candidate_pairs([{"id": "C1", "text": "one", "sources": []}]) == []
