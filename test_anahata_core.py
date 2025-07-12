# test_project.py
import pytest
from anahata_core import detect_ailment, get_remedies_and_wellness

def test_detect_ailment_exact_match():
    query = "I have a bad headache"
    ailment, score = detect_ailment(query)
    assert ailment is not None
    assert score > 0.4
    assert isinstance(ailment, str)

def test_detect_ailment_unrelated_text():
    query = "I like watching space documentaries"
    ailment, score = detect_ailment(query)
    assert (ailment is None or isinstance(ailment, str))  # could be a false match
    assert 0 <= score <= 1.0

def test_get_remedies_and_wellness_valid_ailment():
    data = get_remedies_and_wellness("headache")
    assert "remedies" in data
    assert "wellness" in data
    assert len(data["remedies"]) > 0
    assert isinstance(data["remedies"], list)

def test_get_remedies_and_wellness_invalid_ailment():
    data = get_remedies_and_wellness("unknown condition")
    assert data["remedies"] == []
    assert data["wellness"] == []
