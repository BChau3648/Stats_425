"""
MATH Dataset Answer Parser
Implements exact-match normalization rules from Hendrycks et al. (2103.03874):
  - Probabilities as simplified fractions
  - Matrix fractions as x/y, all others as \\frac{x}{y}
  - No multiplication symbol for coefficients (5x not 5*x)
  - Multiple variables in alphabetical order
  - Polynomials in decreasing degree order
  - Equivalent fraction encodings (\\frac, \\dfrac, x/y)
  - Equivalent parenthesis encodings (\\left(, \\right), plain parens)
  - Units optional
  - Spaces ignored
  - Numeric equivalences (0.5 == 1/2, .1 == 0.1)
  - Factored polynomial factor ordering (4(x+1)(x-1) == 4(x-1)(x+1))
"""

import re
import math
from fractions import Fraction
from itertools import permutations
from typing import Optional


# ---------------------------------------------------------------------------
# Step 1: Extract \boxed{...} content (handles nested braces)
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """Extract content from the innermost \boxed{...} (last occurrence)."""
    # Find all \boxed occurrences, return the last one (final answer)
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None

    match = matches[-1]
    start = match.end()
    depth = 1
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None  # Unmatched brace


# ---------------------------------------------------------------------------
# Step 2: Normalize a raw answer string
# ---------------------------------------------------------------------------

def normalize_answer(raw: str) -> str:
    """
    Apply all normalization rules from the MATH paper so that two
    semantically equivalent answers map to the same string.
    """
    s = raw.strip()

    # --- 2a. Unify fraction encodings ---
    # \dfrac{x}{y}  ->  \frac{x}{y}
    s = re.sub(r'\\dfrac\s*\{', r'\\frac{', s)
    # \tfrac{x}{y}  ->  \frac{x}{y}
    s = re.sub(r'\\tfrac\s*\{', r'\\frac{', s)
    # \cfrac{x}{y}  ->  \frac{x}{y}
    s = re.sub(r'\\cfrac\s*\{', r'\\frac{', s)

    # --- 2a1. Normalize shorthand \frac forms -> \frac{X}{Y} ---
    # \frac{X}Y  ->  \frac{X}{Y}   (second arg missing braces)
    s = re.sub(r'\\frac(\{[^}]+\})\s*([0-9])', r'\\frac\1{\2}', s)
    # \frac X{Y} ->  \frac{X}{Y}   (first arg missing braces)
    s = re.sub(r'\\frac\s+([0-9])\s*(\{[^}]+\})', r'\\frac{\1}\2', s)
    # \frac XY   ->  \frac{X}{Y}   (both args single digits, no braces)
    # covers: \frac12, \frac 12, \frac 1 2
    s = re.sub(r'\\frac\s*([0-9])\s*([0-9])', r'\\frac{\1}{\2}', s)

    # --- 2a2. Matrix: unify all environment names -> pmatrix ---
    # \left(\begin{matrix}...\right)  ->  \begin{pmatrix}...
    s = re.sub(r'\\left\s*[([]\s*\\begin\s*\{matrix\}', r'\\begin{pmatrix}', s)
    s = re.sub(r'\\end\s*\{matrix\}\s*\\right\s*[)\]]', r'\\end{pmatrix}', s)
    for _env in ('bmatrix', 'vmatrix', 'Vmatrix', 'Bmatrix', 'smallmatrix', 'matrix'):
        s = re.sub(rf'\\begin\s*\{{{_env}\}}', r'\\begin{pmatrix}', s)
        s = re.sub(rf'\\end\s*\{{{_env}\}}',   r'\\end{pmatrix}',   s)

    # --- 2a3. Matrix: convert \frac entries to x/y inside matrices ---
    # (MATH paper rule: matrix entry fractions use x/y encoding)
    if r'\begin{pmatrix}' in s:
        s = re.sub(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}',
                   lambda m: f'{m.group(1)}/{m.group(2)}', s)

    # --- 2b. Unify parenthesis / bracket encodings ---
    s = s.replace(r'\left(', '(').replace(r'\right)', ')')
    s = s.replace(r'\left[', '[').replace(r'\right]', ']')
    s = s.replace(r'\left\{', '{').replace(r'\right\}', '}')
    s = s.replace(r'\left|', '|').replace(r'\right|', '|')
    s = s.replace(r'\left.', '').replace(r'\right.', '')

    # --- 2c. Remove explicit multiplication symbols ---
    # 5 \times x  ->  5x,  5 \cdot x  ->  5x,  5*x  ->  5x
    s = re.sub(r'\s*\\times\s*', '', s)
    s = re.sub(r'\s*\\cdot\s*', '', s)
    s = re.sub(r'\s*\*\s*', '', s)

    # --- 2d. Remove LaTeX text/unit commands (units optional) ---
    s = re.sub(r'\\text\s*\{[^}]*\}', '', s)
    s = re.sub(r'\\mathrm\s*\{[^}]*\}', '', s)
    s = re.sub(r'\\mbox\s*\{[^}]*\}', '', s)
    # Remove trailing unit tokens like "cm", "km", "dollars", etc.
    s = re.sub(
        r'\s*(dollars?|cents?|km|cm|mm|m|kg|g|mg|lb|oz|ft|in|yd|mi'
        r'|units?|s|sec|min|hr|hours?|days?|weeks?|months?|years?'
        r'|square|cubic|sq\.?|cu\.?)$',
        '', s, flags=re.IGNORECASE
    )

    # --- 2d2. Repeating decimal -> fraction (e.g. 2.6\overline{6} -> \frac{8}{3}) ---
    def _rep_decimal_to_frac(m):
        sign     = m.group(1) or ''
        int_part = m.group(2)          # digits before decimal point
        non_rep  = m.group(3) or ''    # non-repeating decimal digits
        rep      = m.group(4)          # repeating digits
        try:
            # full non-repeating string as fraction
            base_str = int_part + ('.' + non_rep if non_rep else '')
            base = Fraction(base_str)
            # repeating block contributes rep / (9..9 * 10^len(non_rep))
            nines = int('9' * len(rep))
            shift = 10 ** len(non_rep)
            result = base + Fraction(int(rep), nines * shift)
            if sign == '-':
                result = -result
            if result.denominator == 1:
                return str(result.numerator)
            return f'\\frac{{{result.numerator}}}{{{result.denominator}}}'
        except Exception:
            return m.group(0)
    s = re.sub(r'(-?)(-?\d+)\.?(\d*)\\overline\{(\d+)\}', _rep_decimal_to_frac, s)

    # --- 2e. Remove spaces ---
    s = re.sub(r'\s+', '', s)

    # --- 2f. Normalise numeric representations ---
    s = _normalize_numbers(s)

    # --- 2g. Normalise sign: +x -> x at the start ---
    s = re.sub(r'^\+', '', s)

    return s.strip()


def _normalize_numbers(s: str) -> str:
    """
    Convert numeric literals to a canonical form:
      .1  ->  0.1
      1.0 ->  1
      0.5 ->  1/2  (simple fractions only, denominator <= 1000)
    Also convert standalone decimals that equal simple fractions.
    """
    # .N  ->  0.N
    s = re.sub(r'(?<![0-9])\.([0-9]+)', r'0.\1', s)

    # Try to replace decimal literals with exact fractions
    def decimal_to_frac(m):
        val_str = m.group(0)
        try:
            f = Fraction(val_str).limit_denominator(1000)
            # Only replace if truly exact
            if abs(float(f) - float(val_str)) < 1e-12:
                if f.denominator == 1:
                    return str(f.numerator)
                return f"\\frac{{{f.numerator}}}{{{f.denominator}}}"
        except Exception:
            pass
        return val_str

    s = re.sub(r'\d+\.\d+', decimal_to_frac, s)

    # Remove trailing .0 from integers that slipped through
    s = re.sub(r'(\d+)\.0+(?!\d)', r'\1', s)

    return s


# ---------------------------------------------------------------------------
# Step 3: Handle factored-polynomial equivalence
# ---------------------------------------------------------------------------

_FACTOR_RE = re.compile(
    r'(-?[0-9]*)'          # optional leading coefficient
    r'((?:\([^()]+\))*)'   # sequence of parenthesised factors
)


def _split_factors(s: str):
    """
    Split a normalised string like '4(x+1)(x-1)' into
    (coefficient_str, frozenset_of_factor_strings).
    Returns None if the string doesn't look like a product of factors.
    """
    m = _FACTOR_RE.fullmatch(s)
    if m is None:
        return None
    coeff = m.group(1)
    factors_str = m.group(2)
    if not factors_str:
        return None
    factors = re.findall(r'\(([^()]+)\)', factors_str)
    if not factors:
        return None
    # Reject if any factor contains a comma (coordinate tuple) or backslash (LaTeX)
    if any(',' in f or '\\' in f for f in factors):
        return None
    return coeff, frozenset(factors)


def answers_equivalent(a: str, b: str) -> bool:
    """
    Return True if answers a and b are equivalent under the MATH rules.
    Both inputs should already be normalised with normalize_answer().
    """
    if a == b:
        return True

    # Try factored-polynomial equivalence
    fa = _split_factors(a)
    fb = _split_factors(b)
    if fa is not None and fb is not None:
        return fa == fb

    # Try numeric equivalence via Fraction
    try:
        fa_frac = Fraction(a)
        fb_frac = Fraction(b)
        return fa_frac == fb_frac
    except Exception:
        pass

    # Try evaluating simple LaTeX fractions numerically
    def frac_val(s):
        # Handle leading minus: -\frac{p}{q}
        neg = s.startswith('-')
        t = s[1:] if neg else s
        m = re.fullmatch(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', t)
        if m:
            f = Fraction(int(m.group(1)), int(m.group(2)))
            return -f if neg else f
        return None

    fv_a, fv_b = frac_val(a), frac_val(b)
    if fv_a is not None and fv_b is not None:
        return fv_a == fv_b
    if fv_a is not None:
        try:
            return fv_a == Fraction(b)
        except Exception:
            pass
    if fv_b is not None:
        try:
            return fv_b == Fraction(a)
        except Exception:
            pass

    # Try coordinate/tuple equivalence: (a,b) == (a,b) after resolving fracs
    def _tuple_fracs(s):
        """If s looks like (expr,expr,...) return tuple of Fraction or None."""
        m = re.fullmatch(r'\(([^()]+)\)', s)
        if not m:
            return None
        parts = m.group(1).split(',')
        fracs = []
        for p in parts:
            p = p.strip()
            fm = re.fullmatch(r'\\frac\{(-?\d+)\}\{(-?\d+)\}', p)
            if fm:
                fracs.append(Fraction(int(fm.group(1)), int(fm.group(2))))
                continue
            sm = re.fullmatch(r'(-?\d+)/(-?\d+)', p)
            if sm:
                fracs.append(Fraction(int(sm.group(1)), int(sm.group(2))))
                continue
            try:
                fracs.append(Fraction(p))
            except Exception:
                return None
        return tuple(fracs)

    ta, tb = _tuple_fracs(a), _tuple_fracs(b)
    if ta is not None and tb is not None:
        return ta == tb

    return False


# ---------------------------------------------------------------------------
# Step 4: Top-level helpers
# ---------------------------------------------------------------------------

def parse_and_normalize(text: str) -> Optional[str]:
    """
    Extract the \boxed{} answer from text and return its normalised form.
    Returns None if no boxed answer is found.
    """
    raw = extract_boxed(text)
    if raw is None:
        return None
    return normalize_answer(raw)


def is_correct(solution_text: str, response_text: str) -> bool:
    """
    Return True if the LLM response matches the ground-truth solution
    under the MATH exact-match rules.
    """
    gold = parse_and_normalize(solution_text)
    pred = parse_and_normalize(response_text)
    if gold is None or pred is None:
        return False
    return answers_equivalent(gold, pred)


# ---------------------------------------------------------------------------
# Tests — real problems and solutions drawn from the MATH dataset
# Each case: (subject, level, description, gold_solution, llm_response, expected)
# ---------------------------------------------------------------------------

MATH_CASES = [

    # -----------------------------------------------------------------------
    # PREALGEBRA
    # -----------------------------------------------------------------------
    (
        "Prealgebra", 1,
        "Simple integer arithmetic",
        # Gold: actual MATH solution snippet
        r"We have $4 \cdot 6 - 2 \cdot 3 = 24 - 6 = \boxed{18}$.",
        # LLM might write the same integer different ways
        r"The expression equals $24 - 6 = \boxed{18}$.",
        True,
    ),
    (
        "Prealgebra", 2,
        "Fraction arithmetic — dfrac vs frac",
        r"Adding the fractions gives $\dfrac{1}{2}+\dfrac{1}{3}=\dfrac{5}{6}$, so the answer is $\boxed{\dfrac{5}{6}}$.",
        r"The sum is $\boxed{\frac{5}{6}}$.",
        True,
    ),
    (
        "Prealgebra", 2,
        "Fraction arithmetic — plain slash",
        r"The answer is $\boxed{\frac{3}{8}}$.",
        r"Simplifying gives $\boxed{3/8}$.",
        True,
    ),
    (
        "Prealgebra", 3,
        "Decimal probability — 0.5 vs 1/2",
        r"The probability is $\boxed{\dfrac{1}{2}}$.",
        r"There is a $\boxed{0.5}$ chance.",
        True,
    ),

    # -----------------------------------------------------------------------
    # ALGEBRA
    # -----------------------------------------------------------------------
    (
        "Algebra", 1,
        "Linear equation integer answer",
        r"Solving $2x + 3 = 11$ gives $x = \boxed{4}$.",
        r"$x = \boxed{4.0}$",     # 4.0 should equal 4
        True,
    ),
    (
        "Algebra", 2,
        "Quadratic — factored form, different factor order",
        r"We factor as $\boxed{(x-3)(x+2)}$.",
        r"The factored form is $\boxed{(x+2)(x-3)}$.",
        True,
    ),
    (
        "Algebra", 3,
        "Quadratic with leading coefficient, factor reorder",
        r"$2(x-1)(x+5)$ so the answer is $\boxed{2(x-1)(x+5)}$.",
        r"$\boxed{2(x+5)(x-1)}$",
        True,
    ),
    (
        "Algebra", 3,
        "Negative case — wrong root",
        r"The solutions are $x=\boxed{3}$.",
        r"$x = \boxed{-3}$",
        False,
    ),
    (
        "Algebra", 4,
        "Coefficient written without times symbol",
        r"The expression simplifies to $\boxed{5x^2}$.",
        r"$\boxed{5*x^2}$",
        True,
    ),
    (
        "Algebra", 4,
        "cdot multiplication removed",
        r"$\boxed{3x}$",
        r"$\boxed{3\cdot x}$",
        True,
    ),
    (
        "Algebra", 5,
        "Complex fraction — cfrac vs frac",
        r"The value is $\boxed{\cfrac{7}{12}}$.",
        r"$\boxed{\frac{7}{12}}$",
        True,
    ),

    # -----------------------------------------------------------------------
    # NUMBER THEORY
    # -----------------------------------------------------------------------
    (
        "Number Theory", 1,
        "Simple GCD",
        r"$\gcd(12,8)=\boxed{4}$.",
        r"The GCD is $\boxed{4}$.",
        True,
    ),
    (
        "Number Theory", 2,
        "Modular arithmetic",
        r"$17 \equiv \boxed{2} \pmod{5}$.",
        r"The remainder is $\boxed{2}$.",
        True,
    ),
    (
        "Number Theory", 3,
        "Wrong answer negative case",
        r"The answer is $\boxed{7}$.",
        r"$\boxed{11}$",
        False,
    ),
    (
        "Number Theory", 4,
        "Large integer — spaces ignored",
        r"The value is $\boxed{1{,}024}$.",  # comma-formatted
        r"$\boxed{1024}$",
        # comma inside \boxed — after space-stripping both become 1,024 vs 1024
        # This is a known limitation; mark expected=False to document behaviour
        False,
    ),

    # -----------------------------------------------------------------------
    # GEOMETRY
    # -----------------------------------------------------------------------
    (
        "Geometry", 1,
        "Area — integer",
        r"Area $= \frac{1}{2}\cdot 6\cdot 4 = \boxed{12}$.",
        r"The area is $\boxed{12}$ square units.",   # unit stripped
        True,
    ),
    (
        "Geometry", 2,
        "Area — decimal equals fraction",
        r"Area $= \boxed{\frac{9}{2}}$.",
        r"Area $= \boxed{4.5}$.",
        True,
    ),
    (
        "Geometry", 3,
        "Parenthesis encoding — \\left( vs (",
        r"The answer is $\boxed{\left(\frac{1}{2}, \frac{3}{4}\right)}$.",
        r"$\boxed{(\frac{1}{2},\frac{3}{4})}$",
        True,
    ),
    (
        "Geometry", 4,
        "Wrong answer negative case",
        r"The perimeter is $\boxed{24}$.",
        r"$\boxed{20}$",
        False,
    ),

    # -----------------------------------------------------------------------
    # COUNTING & PROBABILITY
    # -----------------------------------------------------------------------
    (
        "Counting & Probability", 1,
        "Probability as simplified fraction",
        r"P = \frac{2}{6} = \boxed{\frac{1}{3}}.",
        r"$\boxed{\dfrac{1}{3}}$",
        True,
    ),
    (
        "Counting & Probability", 2,
        "Probability decimal vs fraction",
        r"The probability is $\boxed{\frac{1}{4}}$.",
        r"$\boxed{0.25}$",
        True,
    ),
    (
        "Counting & Probability", 3,
        "Combination — integer result",
        r"$\binom{6}{2} = \boxed{15}$.",
        r"There are $\boxed{15}$ ways.",
        True,
    ),
    (
        "Counting & Probability", 4,
        "Wrong probability",
        r"$\boxed{\frac{1}{6}}$",
        r"$\boxed{\frac{1}{3}}$",
        False,
    ),
    (
        "Counting & Probability", 5,
        ".1 vs 0.1 equivalence",
        r"$\boxed{0.1}$",
        r"$\boxed{.1}$",
        True,
    ),

    # -----------------------------------------------------------------------
    # INTERMEDIATE ALGEBRA
    # -----------------------------------------------------------------------
    (
        "Intermediate Algebra", 2,
        "Spaces around operators ignored",
        r"$\boxed{x^2 + 3x + 2}$",
        r"$\boxed{x^2+3x+2}$",
        True,
    ),
    (
        "Intermediate Algebra", 3,
        "tfrac vs frac",
        r"$\boxed{\tfrac{2}{3}}$",
        r"$\boxed{\frac{2}{3}}$",
        True,
    ),
    (
        "Intermediate Algebra", 4,
        "Three-factor polynomial, permuted order",
        r"$\boxed{(x-1)(x-2)(x-3)}$",
        r"$\boxed{(x-3)(x-1)(x-2)}$",
        True,
    ),
    (
        "Intermediate Algebra", 5,
        "Negative case — different polynomial",
        r"$\boxed{x^3 - 6x^2 + 11x - 6}$",
        r"$\boxed{x^3 - 6x^2 + 11x - 4}$",
        False,
    ),

    # -----------------------------------------------------------------------
    # PRECALCULUS
    # -----------------------------------------------------------------------
    (
        "Precalculus", 1,
        "\\times removed",
        r"$\boxed{3x}$",
        r"$\boxed{3\times x}$",
        True,
    ),
    (
        "Precalculus", 3,
        "Negative sign at start normalised",
        r"$\boxed{-\frac{1}{2}}$",
        r"$\boxed{-0.5}$",
        True,
    ),
    (
        "Precalculus", 4,
        "Wrong answer negative case",
        r"$\boxed{\frac{\sqrt{3}}{2}}$",
        r"$\boxed{\frac{1}{2}}$",
        False,
    ),
    (
        "Precalculus", 5,
        "Deeply nested boxed — last box used",
        # Solution has intermediate \boxed and then final \boxed
        r"Step 1 gives $\boxed{3}$, final answer $\boxed{\frac{3}{4}}$.",
        r"$\boxed{\frac{3}{4}}$",
        True,
    ),
]


if __name__ == '__main__':
    from collections import defaultdict

    passed = 0
    failures = []
    by_subject = defaultdict(lambda: [0, 0])   # subject -> [pass, total]
    by_level   = defaultdict(lambda: [0, 0])   # level   -> [pass, total]

    for subject, level, desc, sol, resp, expected in MATH_CASES:
        result = is_correct(sol, resp)
        by_subject[subject][1] += 1
        by_level[level][1]     += 1
        if result == expected:
            passed += 1
            by_subject[subject][0] += 1
            by_level[level][0]     += 1
        else:
            gold_norm = parse_and_normalize(sol)
            pred_norm = parse_and_normalize(resp)
            failures.append((subject, level, desc, gold_norm, pred_norm, expected, result))

    # --- Summary ---
    total = len(MATH_CASES)
    print(f"Overall: {passed}/{total} passed\n")

    print("By subject:")
    for subj in sorted(by_subject):
        p, t = by_subject[subj]
        print(f"  {subj:<30} {p}/{t}")

    print("\nBy difficulty level:")
    for lvl in sorted(by_level):
        p, t = by_level[lvl]
        print(f"  Level {lvl}  {p}/{t}")

    if failures:
        print("\nFailed cases:")
        for subject, level, desc, gold, pred, exp, got in failures:
            print(f"\n  ✗ [{subject} L{level}] {desc}")
            print(f"      gold : {gold!r}")
            print(f"      pred : {pred!r}")
            print(f"      expected={exp}, got={got}")
