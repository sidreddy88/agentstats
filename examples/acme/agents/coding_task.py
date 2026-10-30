"""
Acme's coding agent, as an Inspect AI task: fix a small buggy Python function; hidden tests decide.

    inspect eval agents/coding_task.py --model together/deepseek-ai/DeepSeek-V4.1-Flash --epochs 2 \\
        --log-dir data/coding --log-format json

The scorer runs the hidden tests in a subprocess with a timeout. agentstats reads the log directly
(`agentstats summary data/coding/*.json`); sample metadata `kind` becomes a tag, so results slice by kind.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate, system_message

BUGS = [
    ("strings", "def is_palindrome(s):\n    return s == s[::-1]",
     "Ignore case and non-alphanumeric characters.",
     "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('abc')"),
    ("strings", "def word_count(text):\n    return len(text.split(' '))",
     "Count words separated by any whitespace; empty text has 0 words.",
     "assert word_count('') == 0\nassert word_count('a  b\\tc\\n') == 3"),
    ("strings", "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())",
     "Keep the original spacing between words.",
     "assert capitalize_words('hello  world') == 'Hello  World'"),
    ("strings", "def reverse_words(s):\n    return ' '.join(s.split(' ')[::-1])",
     "Reverse word order; collapse repeated spaces and strip the ends.",
     "assert reverse_words('  a  b c ') == 'c b a'"),
    ("lists", "def second_largest(xs):\n    return sorted(xs)[-2]",
     "Return the second largest distinct value, or None if there isn't one.",
     "assert second_largest([3, 3, 1]) == 1\nassert second_largest([5]) is None\n"
     "assert second_largest([2, 2]) is None"),
    ("lists", "def chunk(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n, n)]",
     "Split into chunks of size n; the last chunk may be shorter.",
     "assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\nassert chunk([], 3) == []"),
    ("lists", "def dedupe(xs):\n    return list(set(xs))",
     "Remove duplicates, keeping the first occurrence order.",
     "assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]"),
    ("lists", "def flatten(xs):\n    return [y for x in xs for y in x]",
     "Flatten arbitrarily nested lists.",
     "assert flatten([1, [2, [3, [4]]], 5]) == [1, 2, 3, 4, 5]"),
    ("math", "def mean(xs):\n    return sum(xs) / len(xs)",
     "Return None for an empty list.",
     "assert mean([]) is None\nassert mean([1, 2, 3]) == 2"),
    ("math", "def is_prime(n):\n    return all(n % d for d in range(2, n))",
     "0 and 1 are not prime; be correct for negative numbers too.",
     "assert not is_prime(1) and not is_prime(0) and not is_prime(-7)\nassert is_prime(2) and is_prime(97)"),
    ("math", "def median(xs):\n    xs = sorted(xs)\n    return xs[len(xs) // 2]",
     "For an even count, return the mean of the two middle values.",
     "assert median([4, 1, 3, 2]) == 2.5\nassert median([3, 1, 2]) == 2"),
    ("math", "def round_half_up(x):\n    return round(x)",
     "Round .5 away from zero (2.5 -> 3, -2.5 -> -3), unlike Python's banker's rounding.",
     "assert round_half_up(2.5) == 3 and round_half_up(-2.5) == -3 and round_half_up(1.4) == 1"),
]

SYSTEM = ("Fix the bug in the Python function so it meets the requirement. Reply with the complete corrected "
          "function in one ```python code block and nothing else.")


def samples() -> list[Sample]:
    return [Sample(id=f"bug-{i:02d}", input=f"Requirement: {req}\n\n```python\n{code}\n```",
                   target=tests, metadata={"kind": kind})
            for i, (kind, code, req, tests) in enumerate(BUGS)]


@scorer(metrics=[accuracy()])
def hidden_tests(timeout: int = 10):
    async def score(state: TaskState, target: Target) -> Score:
        m = re.search(r"```(?:python)?\n(.*?)```", state.output.completion or "", re.S)
        if not m:
            return Score(value=INCORRECT, explanation="no code block")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(m.group(1) + "\n\n" + target.text + "\n")
        try:
            r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return Score(value=INCORRECT, explanation="timeout")
        ok = r.returncode == 0
        return Score(value=CORRECT if ok else INCORRECT, explanation=r.stderr.decode()[-300:])
    return score


@task
def fix_bugs():
    return Task(dataset=samples(), solver=[system_message(SYSTEM), generate()], scorer=hidden_tests())
