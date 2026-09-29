"""Numerical definitions used in the saved-run analyses; standard library only."""
import math
import statistics


def distribution_summary(values):
    """Summary using the nearest-rank P95; an empty collection is undefined."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError('Expected at least one observation')
    return dict(n=len(ordered), minimum=ordered[0], median=statistics.median(ordered),
                mean=statistics.mean(ordered), p95=ordered[math.ceil(.95*len(ordered))-1],
                maximum=ordered[-1], total=sum(ordered))


def boxplot_summary(values):
    """Quartiles interpolated at (n-1)*p; whiskers at observed values within 1.5 IQR."""
    ordered = sorted(values)
    if len(ordered) < 2:
        raise ValueError('Expected at least two observations')
    q1, median, q3 = statistics.quantiles(ordered, n=4, method='inclusive')
    iqr = q3-q1
    inside = [v for v in ordered if q1-1.5*iqr <= v <= q3+1.5*iqr]
    return dict(n=len(ordered), q1=q1, median=median, q3=q3,
                lower_whisker=inside[0], upper_whisker=inside[-1],
                outliers=[v for v in ordered if v < inside[0] or v > inside[-1]])


def wilson95(successes, total):
    """Two-sided 95% Wilson score interval; undefined for an empty denominator."""
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError('Expected 0 <= successes <= total and total > 0')
    p = successes / total
    z = 1.959963984540054
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    delta = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0.0, center-delta), min(1.0, center+delta)]


def mcnemar_exact(first_only, second_only):
    """Conditional two-sided binomial McNemar p-value, without approximation."""
    if any(not isinstance(x, int) or x < 0 for x in (first_only, second_only)):
        raise ValueError('Discordant counts must be nonnegative integers')
    n = first_only + second_only
    return min(1.0, 2*sum(math.comb(n, k) for k in range(min(first_only, second_only)+1))/2**n)


def retrieval_at_k(retrieved_ids, reference_ids, k=3):
    """Recall and reciprocal rank for one case; None means no reference label."""
    if k <= 0:
        raise ValueError('k must be positive')
    reference = set(reference_ids)
    if not reference:
        return None
    retrieved = list(retrieved_ids)[:k]
    if len(set(retrieved)) != len(retrieved):
        raise ValueError('Duplicate retrieved identifiers within cutoff')
    first_rank = next((i for i, cid in enumerate(retrieved, 1) if cid in reference), None)
    return dict(recall=len(set(retrieved) & reference)/len(reference),
                reciprocal_rank=1/first_rank if first_rank else 0.0,
                first_reference_rank=first_rank)
