import random
import string

import numpy as np
import pytest

from mcp_shield.winnow import (
    GUARANTEE,
    Fingerprints,
    find_hits,
    fingerprint,
    gram_hashes,
    longest_shared_run,
    winnow,
)


def rand_text(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(string.ascii_lowercase + "     ") for _ in range(n))


def test_density_about_one_in_eleven():
    rng = random.Random(1)
    text = rand_text(rng, 50_000)
    fp = winnow(text)
    density = len(fp) / (len(text) - 11)
    assert 0.07 < density < 0.11  # 2 / (w + 1) = 0.0909


@pytest.mark.parametrize("seed", range(25))
def test_guarantee_any_32_char_shared_run_is_found(seed):
    rng = random.Random(seed)
    secret = rand_text(rng, GUARANTEE).replace(" ", "x")
    record = rand_text(rng, rng.randint(0, 4000)) + secret + rand_text(rng, rng.randint(0, 4000))
    query = rand_text(rng, rng.randint(0, 300)) + secret + rand_text(rng, rng.randint(0, 300))
    canon_r, fp = fingerprint(record)
    canon_q, _ = fingerprint(query)
    assert find_hits(gram_hashes(canon_q), fp), "winnowing must catch every shared run of >= 32 characters"
    assert longest_shared_run(canon_q, canon_r, fp) >= GUARANTEE


def test_short_common_phrase_is_not_a_flow():
    record_c, fp = fingerprint("The quarterly report shows that the information is accurate and complete.")
    query_c, _ = fingerprint("Please write that the information is fine.")
    assert longest_shared_run(query_c, record_c, fp) < GUARANTEE


def test_without_record_text_any_hit_counts():
    _record_c, fp = fingerprint("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo")
    query_c, _ = fingerprint("xx charlie delta echo foxtrot yy")
    assert longest_shared_run(query_c, None, fp) >= GUARANTEE  # conservative after a restart


def test_canonicalisation_defeats_case_and_spacing():
    record_c, fp = fingerprint("AWS_SECRET=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY   end")
    query_c, _ = fingerprint("aws_secret=wjalrxutnfemi/k7mdeng/bpxrficyexamplekey")
    assert longest_shared_run(query_c, record_c, fp) >= GUARANTEE


def test_serialisation_round_trip():
    _, fp = fingerprint("some reasonably long text for fingerprinting purposes here")
    back = Fingerprints.from_bytes(fp.to_bytes())
    assert np.array_equal(back.hashes, fp.hashes) and np.array_equal(back.positions, fp.positions)


def test_tiny_inputs():
    assert len(winnow("short")) == 0
    assert len(winnow("exactly twelve")) >= 1


# Real documents repeat phrases, so the same 12-gram is often kept at several positions.
# Verification has to measure the run at every one of them, not only the first.
PHRASES = [
    "the server ",
    "mcp shield ",
    "tool result ",
    "policy gate ",
    "taint record ",
    "the host ",
    "every call ",
    "is blocked ",
    "and the ",
    "session key ",
    "read file ",
    "of the ",
]


def repetitive_text(rng: random.Random, n: int) -> str:
    out = ""
    while len(out) < n:
        out += rng.choice(PHRASES)
    return out[:n]


def test_guarantee_holds_when_kept_grams_repeat():
    rng = random.Random(7)
    misses = 0
    repeated = 0
    for _ in range(300):
        canon_r, fp = fingerprint(repetitive_text(rng, 4096))
        repeated += int(np.unique(fp.hashes).size < fp.hashes.size)
        pos = rng.randrange(1, len(canon_r) - GUARANTEE - 1)
        secret = canon_r[pos : pos + GUARANTEE]
        query = "0123456789" + secret + "9876543210"  # digits never occur in the record
        canon_q, _ = fingerprint(query)
        misses += longest_shared_run(canon_q, canon_r, fp) < GUARANTEE
    assert misses == 0, f"{misses} of 300 planted 32-character runs were missed"
    assert repeated > 0, "the test needs records whose kept grams repeat"


def test_find_hits_reports_every_kept_occurrence():
    rng = random.Random(3)
    _, fp = fingerprint(repetitive_text(rng, 4096))
    values, counts = np.unique(fp.hashes, return_counts=True)
    dup = values[counts > 1][0]
    hits = find_hits(np.array([dup], dtype=np.uint64), fp)
    assert sorted(r for _, r in hits) == sorted(int(p) for p in fp.positions[fp.hashes == dup])
