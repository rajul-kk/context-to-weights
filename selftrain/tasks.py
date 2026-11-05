import random
import re
from dataclasses import dataclass

FAMILIES = ("add", "mul", "sort", "even", "count")
SUFFIX = "Think briefly, then finish with a final line of the form 'Answer: <answer>'."
_ANSWER = re.compile(r"answer\s*[:=]\s*(.+)", re.IGNORECASE)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_INT = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Task:
    family: str
    question: str
    answer: str

    @property
    def prompt(self):
        return f"{self.question}\n{SUFFIX}"


def _add(rng):
    a, b = rng.randint(100000, 999999), rng.randint(100000, 999999)
    return Task("add", f"What is {a} + {b}?", str(a + b))


def _mul(rng):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    return Task("mul", f"What is {a} * {b}?", str(a * b))


def _sort(rng):
    values = [rng.randint(1, 99) for _ in range(6)]
    listed = ", ".join(map(str, values))
    return Task("sort", f"Sort these numbers in ascending order: {listed}.",
                ", ".join(map(str, sorted(values))))


def _even(rng):
    values = [rng.randint(1, 99) for _ in range(8)]
    listed = ", ".join(map(str, values))
    return Task("even", f"How many of these numbers are even: {listed}?",
                str(sum(v % 2 == 0 for v in values)))


def _count(rng):
    letters = "".join(rng.choice("abc") for _ in range(8))
    target = rng.choice("abc")
    return Task("count", f"How many times does the letter '{target}' appear in '{letters}'?",
                str(letters.count(target)))


BUILDERS = {"add": _add, "mul": _mul, "sort": _sort, "even": _even, "count": _count}


def make_tasks(n, seed, families=FAMILIES):
    rng = random.Random(seed)
    families = tuple(families)
    return [BUILDERS[families[i % len(families)]](rng) for i in range(n)]


def extract(text):
    found = _ANSWER.findall(text or "")
    if not found:
        return None
    return found[-1].strip().strip("*").strip().rstrip(".").strip()


def _ints(value):
    return [int(m) for m in _INT.findall(_THOUSANDS.sub("", value))]


def check(task, text):
    given = extract(text)
    if given is None:
        return False
    if task.family == "sort":
        return _ints(given) == _ints(task.answer)
    parsed = _ints(given)
    return bool(parsed) and parsed[0] == int(task.answer)
