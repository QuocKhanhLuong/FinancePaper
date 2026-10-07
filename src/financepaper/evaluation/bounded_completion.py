"""Sharp finite-domain bounds for one missing SHAP player (research result)."""
from fractions import Fraction
from math import comb, factorial

def scalar_width(m, p):
    """Rational closed form for the sharp scalar oscillation, homogeneous P(A)."""
    p = Fraction(p)
    if not isinstance(m, int) or m < 1 or not 0 < p < 1:
        raise ValueError("m>=1 and 0<p<1 required")
    weight = (1-p)*sum((Fraction(comb(m-1, k), k+2)*p**(m-1-k)*(1-p)**k
                       for k in range(m)), Fraction(0))
    return 1-p-weight


def completion_kernel(m, p):
    """Exact K_i(a); includes the constant all-observed-one column."""
    p = Fraction(p)
    scalar_width(m, p)  # validate
    result = []
    for i in range(m):
        row = []
        for code in range(2**m):
            polynomial = [Fraction(1)]
            for j in range(m):
                if j == i:
                    continue
                factor = [p, 1-p] if code & (1 << j) else [1-p, -(1-p)]
                out = [Fraction(0)]*(len(polynomial)+1)
                for a, value in enumerate(polynomial):
                    for b, coefficient in enumerate(factor):
                        out[a+b] += value*coefficient
                polynomial = out
            integral = sum((v/Fraction(k+2) for k, v in enumerate(polynomial)), Fraction(0))
            row.append((1-p)*(1 if code & (1 << i) else -1)*integral)
        result.append(row)
    return result


def definition_matrix(m, p, hidden_value):
    """Direct exact-rational SHAP linear operator from coalition/background sums.

    No integral/kernel formula used. Columns index binary (A,H) truth-table rows;
    low m bits are observed A, bit m is H. P_H is fair; query A=1.
    """
    if hidden_value not in (0, 1):
        raise ValueError("Binary hidden query required")
    return finite_definition_matrix(m, p, [Fraction(1, 2)]*2, hidden_value)


def finite_definition_matrix(m, p, probabilities, hidden_value):
    """Exact definition-level operator, with a general finite hidden marginal."""
    p = Fraction(p)
    scalar_width(m, p)
    q = [Fraction(v) for v in probabilities]
    if not q or min(q) <= 0 or sum(q) != 1 or hidden_value not in range(len(q)):
        raise ValueError("Positive exact finite hidden distribution required")
    d, coalitions, count, observed_count = m+1, 2**(m+1), 2**m*len(q), 2**m
    background = [p**((code % observed_count).bit_count())*(1-p)**(m-(code % observed_count).bit_count())*q[code//observed_count] for code in range(count)]
    values = []
    for coalition in range(coalitions):
        row = [Fraction(0)]*count
        for code, mass in enumerate(background):
            observed_mask = coalition % observed_count
            a = (code % observed_count) | observed_mask
            h = hidden_value if coalition & observed_count else code//observed_count
            hybrid = a+observed_count*h
            row[hybrid] += mass
        values.append(row)
    answer = []
    for i in range(m):
        row = [Fraction(0)]*count
        for coalition in range(coalitions):
            if coalition & (1 << i):
                continue
            size = coalition.bit_count()
            weight = Fraction(factorial(size)*factorial(d-size-1), factorial(d))
            for code in range(count):
                row[code] += weight*(values[coalition | (1 << i)][code]-values[coalition][code])
        answer.append(row)
    return answer


def fixed_law_factor(probabilities):
    """Exact beta(q) and attaining partition; exponential support enumeration.

    This is a small-support oracle, not a new efficient partition algorithm.
    """
    q = [Fraction(v) for v in probabilities]
    if not q or min(q) <= 0 or sum(q) != 1:
        raise ValueError("Positive normalized exact masses required")
    best, indices = Fraction(-1), ()
    for code in range(2**len(q)):
        selected = tuple(i for i in range(len(q)) if code & (1 << i))
        mass = sum((q[i] for i in selected), Fraction(0))
        value = mass*(1-mass)
        if value > best:
            best, indices = value, selected
    return best, indices


def sharp_tree(m, target=0, constant=.5):
    """A single O(m)-leaf, depth m+1 bounded tree attaining scalar width.

    H=0: output 1 if target A=1 and another observed A=0, else 0;
    H=1: output 1 if target A=0, else 0. Override all-A=1 with c.
    """
    if not isinstance(m, int) or m < 1 or target not in range(m) or not 0 <= constant <= 1:
        raise ValueError("Invalid sharp witness parameters")
    def all_others(h):
        node = float(constant)
        for j in reversed([j for j in range(m) if j != target]):
            node = dict(feature=j, threshold=.5, left=float(1-h), right=node)
        return node
    return dict(feature=m, threshold=.5,
                left=dict(feature=target, threshold=.5, left=0., right=all_others(0)),
                right=dict(feature=target, threshold=.5, left=1., right=all_others(1)))


def directional_width(kernel, direction):
    """Exact support width of the image of the unconstrained truth-table cube."""
    if len(direction) != len(kernel):
        raise ValueError("Direction dimension mismatch")
    return sum((abs(sum((Fraction(v)*kernel[i][code] for i, v in enumerate(direction)), Fraction(0)))
                for code in range(len(kernel[0])-1)), Fraction(0))
