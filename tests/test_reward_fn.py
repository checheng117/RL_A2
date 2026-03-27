from src.rewards.reward_fn import compute_reward, make_trl_reward_fn


def test_v1_requires_tags():
    r = compute_reward("v1", "p", "[point] a\n[reason] 1. b\n[summary] c", "src")
    assert r > 0.5
    r0 = compute_reward("v1", "p", "no tags", "src")
    assert r0 < 0.4


def test_trl_wrapper():
    fn = make_trl_reward_fn("v1")
    out = fn(
        completions=["[point] x\n[reason] 1. a\n[summary] y"],
        prompts=["p"],
        answer_en=["source"],
    )
    assert len(out) == 1
    assert isinstance(out[0], float)


def test_v1_vs_v4_hacking_sanity():
    """Standard good summary; empty tag soup; repeated legal blocks (V1 high, V4 should be lower)."""
    src = "The court sentenced the defendant to seven years after prosecution asked for four."
    good = (
        "[point]The sentence exceeded the prosecution request.\n"
        "[reason]1. The judge imposed seven years. 2. The prosecution had asked for a shorter term.\n"
        "[summary]The outcome shows judicial discretion beyond the prosecution's recommendation."
    )
    tag_soup = "[point]x[reason]y[summary]z"
    dup = (
        "[point]a[reason]1. b 2. c[summary]s"
        "[point]a[reason]1. b 2. c[summary]s"
    )
    g1, g4 = compute_reward("v1", "", good, src), compute_reward("v4", "", good, src)
    assert g1 > 0.8 and g4 > 0.3
    t1, t4 = compute_reward("v1", "", tag_soup, src), compute_reward("v4", "", tag_soup, src)
    assert t1 >= 0.99
    assert t4 < t1
    d1, d4 = compute_reward("v1", "", dup, src), compute_reward("v4", "", dup, src)
    assert d1 >= 0.99
    assert d4 < d1 or d4 < 0.99
