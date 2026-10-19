import random
import re
import string
from dataclasses import dataclass

FAMILIES = ("add", "mul", "sort", "reverse", "count")
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
    a, b = rng.randint(100, 999), rng.randint(100, 999)
    return Task("add", f"What is {a} + {b}?", str(a + b))


def _mul(rng):
    a, b = rng.randint(12, 99), rng.randint(12, 99)
    return Task("mul", f"What is {a} * {b}?", str(a * b))


def _sort(rng):
    values = [rng.randint(1, 99) for _ in range(6)]
    listed = ", ".join(map(str, values))
    return Task("sort", f"Sort these numbers in ascending order: {listed}.",
                ", ".join(map(str, sorted(values))))


def _reverse(rng):
    word = "".join(rng.choice(string.ascii_lowercase) for _ in range(7))
    return Task("reverse", f"Reverse the letters of the string '{word}'.", word[::-1])


def _count(rng):
    letters = "".join(rng.choice("abcde") for _ in range(14))
    target = rng.choice("abcde")
    return Task("count", f"How many times does the letter '{target}' appear in '{letters}'?",
                str(letters.count(target)))


BUILDERS = {"add": _add, "mul": _mul, "sort": _sort, "reverse": _reverse, "count": _count}


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


def _letters(value):
    return "".join(ch for ch in value.lower() if ch.isalpha())


def check(task, text):
    given = extract(text)
    if given is None:
        return False
    if task.family == "sort":
        return _ints(given) == _ints(task.answer)
    if task.family == "reverse":
        return _letters(given) == task.answer
    parsed = _ints(given)
    return bool(parsed) and parsed[0] == int(task.answer)
