

def test_suite_health_kinds():
    from agentstats.reliability import suite_health
    reg = suite_health({f"c{i}": (2, 2) for i in range(93)} | {f"f{i}": (2, 0) for i in range(7)})
    assert reg["kind"] == "regression" and reg["always_share"] == 0.93
    front = suite_health({f"c{i}": (2, 1) for i in range(50)})
    assert front["kind"] == "frontier"
    hard = suite_health({f"c{i}": (2, 0) for i in range(80)} | {f"p{i}": (2, 2) for i in range(20)})
    assert hard["kind"] == "too_hard" and hard["never_share"] == 0.8
    assert suite_health({f"c{i}": (1, 1) for i in range(15)} | {f"f{i}": (1, 0) for i in range(5)})["kind"] == "between"
    assert suite_health({}) is None
