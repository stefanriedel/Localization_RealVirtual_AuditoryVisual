from __future__ import annotations
import math
import warnings
from collections import namedtuple

import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, sqrt, array, atleast_1d,  # noqa: F401
                   compress, pi, exp, ravel, count_nonzero, sin, cos,
                   arctan2, hypot)

from scipy import optimize, special, interpolate, stats
from scipy._lib._bunch import _make_tuple_bunch
from scipy._lib._util import _rename_parameter, _contains_nan, _get_nan

from scipy.stats._ansari_swilk_statistics import gscale, swilk
from scipy.stats import _stats_py
from scipy.stats._fit import FitResult
from scipy.stats._stats_py import find_repeats, _normtest_finish, SignificanceResult
from scipy.stats.contingency import chi2_contingency
from scipy.stats import distributions
from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats._hypotests import _get_wilcoxon_distr
from scipy.stats._axis_nan_policy import _axis_nan_policy_factory

WilcoxonResult = _make_tuple_bunch('WilcoxonResult', ['statistic', 'pvalue'])


def wilcoxon_result_unpacker(res):
    if hasattr(res, 'zstatistic'):
        return res.statistic, res.pvalue, res.zstatistic
    else:
        return res.statistic, res.pvalue


def wilcoxon_result_object(statistic, pvalue, zstatistic=None):
    res = WilcoxonResult(statistic, pvalue)
    if zstatistic is not None:
        res.zstatistic = zstatistic
    return res


def wilcoxon_outputs(kwds):
    method = kwds.get('method', 'auto')
    if method == 'approx':
        return 3
    return 2

def stats_wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided", method='auto'):
    """Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences ``x - y`` is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        Either the first set of measurements (in which case ``y`` is the second
        set of measurements), or the differences between two sets of
        measurements (in which case ``y`` is not to be specified.)  Must be
        one-dimensional.
    y : array_like, optional
        Either the second set of measurements (if ``x`` is the first set of
        measurements), or not specified (if ``x`` is the differences between
        two sets of measurements.)  Must be one-dimensional.

        .. warning::
            When `y` is provided, `wilcoxon` calculates the test statistic
            based on the ranks of the absolute values of ``d = x - y``.
            Roundoff error in the subtraction can result in elements of ``d``
            being assigned different ranks even when they would be tied with
            exact arithmetic. Rather than passing `x` and `y` separately,
            consider computing the difference ``x - y``, rounding as needed to
            ensure that only truly unique elements are numerically distinct,
            and passing the result as `x`, leaving `y` at the default (None).

    zero_method : {"wilcox", "pratt", "zsplit"}, optional
        There are different conventions for handling pairs of observations
        with equal values ("zero-differences", or "zeros").

        * "wilcox": Discards all zero-differences (default); see [4]_.
        * "pratt": Includes zero-differences in the ranking process,
          but drops the ranks of the zeros (more conservative); see [3]_.
          In this case, the normal approximation is adjusted as in [5]_.
        * "zsplit": Includes zero-differences in the ranking process and
          splits the zero rank between positive and negative ones.

    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic if a normal approximation is used.  Default is False.
    alternative : {"two-sided", "greater", "less"}, optional
        Defines the alternative hypothesis. Default is 'two-sided'.
        In the following, let ``d`` represent the difference between the paired
        samples: ``d = x - y`` if both ``x`` and ``y`` are provided, or
        ``d = x`` otherwise.

        * 'two-sided': the distribution underlying ``d`` is not symmetric
          about zero.
        * 'less': the distribution underlying ``d`` is stochastically less
          than a distribution symmetric about zero.
        * 'greater': the distribution underlying ``d`` is stochastically
          greater than a distribution symmetric about zero.

    method : {"auto", "exact", "approx"}, optional
        Method to calculate the p-value, see Notes. Default is "auto".

    Returns
    -------
    An object with the following attributes.

    statistic : array_like
        If `alternative` is "two-sided", the sum of the ranks of the
        differences above or below zero, whichever is smaller.
        Otherwise the sum of the ranks of the differences above zero.
    pvalue : array_like
        The p-value for the test depending on `alternative` and `method`.
    zstatistic : array_like
        When ``method = 'approx'``, this is the normalized z-statistic::

            z = (T - mn - d) / se

        where ``T`` is `statistic` as defined above, ``mn`` is the mean of the
        distribution under the null hypothesis, ``d`` is a continuity
        correction, and ``se`` is the standard error.
        When ``method != 'approx'``, this attribute is not available.

    See Also
    --------
    kruskal, mannwhitneyu

    Notes
    -----
    In the following, let ``d`` represent the difference between the paired
    samples: ``d = x - y`` if both ``x`` and ``y`` are provided, or ``d = x``
    otherwise. Assume that all elements of ``d`` are independent and
    identically distributed observations, and all are distinct and nonzero.

    - When ``len(d)`` is sufficiently large, the null distribution of the
      normalized test statistic (`zstatistic` above) is approximately normal,
      and ``method = 'approx'`` can be used to compute the p-value.

    - When ``len(d)`` is small, the normal approximation may not be accurate,
      and ``method='exact'`` is preferred (at the cost of additional
      execution time).

    - The default, ``method='auto'``, selects between the two: when
      ``len(d) <= 50``, the exact method is used; otherwise, the approximate
      method is used.

    The presence of "ties" (i.e. not all elements of ``d`` are unique) and
    "zeros" (i.e. elements of ``d`` are zero) changes the null distribution
    of the test statistic, and ``method='exact'`` no longer calculates
    the exact p-value. If ``method='approx'``, the z-statistic is adjusted
    for more accurate comparison against the standard normal, but still,
    for finite sample sizes, the standard normal is only an approximation of
    the true null distribution of the z-statistic. There is no clear
    consensus among references on which method most accurately approximates
    the p-value for small samples in the presence of zeros and/or ties. In any
    case, this is the behavior of `wilcoxon` when ``method='auto':
    ``method='exact'`` is used when ``len(d) <= 50`` *and there are no zeros*;
    otherwise, ``method='approx'`` is used.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    .. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
    .. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
       Rank Procedures, Journal of the American Statistical Association,
       Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
    .. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
       Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
    .. [5] Cureton, E.E., The Normal Approximation to the Signed-Rank
       Sampling Distribution When Zero Differences are Present,
       Journal of the American Statistical Association, Vol. 62, 1967,
       pp. 1068-1069. :doi:`10.1080/01621459.1967.10500917`

    Examples
    --------
    In [4]_, the differences in height between cross- and self-fertilized
    corn plants is given as follows:

    >>> d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

    Cross-fertilized plants appear to be higher. To test the null
    hypothesis that there is no height difference, we can apply the
    two-sided test:

    >>> from scipy.stats import wilcoxon
    >>> res = wilcoxon(d)
    >>> res.statistic, res.pvalue
    (24.0, 0.041259765625)

    Hence, we would reject the null hypothesis at a confidence level of 5%,
    concluding that there is a difference in height between the groups.
    To confirm that the median of the differences can be assumed to be
    positive, we use:

    >>> res = wilcoxon(d, alternative='greater')
    >>> res.statistic, res.pvalue
    (96.0, 0.0206298828125)

    This shows that the null hypothesis that the median is negative can be
    rejected at a confidence level of 5% in favor of the alternative that
    the median is greater than zero. The p-values above are exact. Using the
    normal approximation gives very similar values:

    >>> res = wilcoxon(d, method='approx')
    >>> res.statistic, res.pvalue
    (24.0, 0.04088813291185591)

    Note that the statistic changed to 96 in the one-sided case (the sum
    of ranks of positive differences) whereas it is 24 in the two-sided
    case (the minimum of sum of ranks above and below zero).

    In the example above, the differences in height between paired plants are
    provided to `wilcoxon` directly. Alternatively, `wilcoxon` accepts two
    samples of equal length, calculates the differences between paired
    elements, then performs the test. Consider the samples ``x`` and ``y``:

    >>> import numpy as np
    >>> x = np.array([0.5, 0.825, 0.375, 0.5])
    >>> y = np.array([0.525, 0.775, 0.325, 0.55])
    >>> res = wilcoxon(x, y, alternative='greater')
    >>> res
    WilcoxonResult(statistic=5.0, pvalue=0.5625)

    Note that had we calculated the differences by hand, the test would have
    produced different results:

    >>> d = [-0.025, 0.05, 0.05, -0.05]
    >>> ref = wilcoxon(d, alternative='greater')
    >>> ref
    WilcoxonResult(statistic=6.0, pvalue=0.4375)

    The substantial difference is due to roundoff error in the results of
    ``x-y``:

    >>> d - (x-y)
    array([2.08166817e-17, 6.93889390e-17, 1.38777878e-17, 4.16333634e-17])

    Even though we expected all the elements of ``(x-y)[1:]`` to have the same
    magnitude ``0.05``, they have slightly different magnitudes in practice,
    and therefore are assigned different ranks in the test. Before performing
    the test, consider calculating ``d`` and adjusting it as necessary to
    ensure that theoretically identically values are not numerically distinct.
    For example:

    >>> d2 = np.around(x - y, decimals=3)
    >>> wilcoxon(d2, alternative='greater')
    WilcoxonResult(statistic=6.0, pvalue=0.4375)

    """
    mode = method

    if mode not in ["auto", "approx", "exact"]:
        raise ValueError("mode must be either 'auto', 'approx' or 'exact'")

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method must be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError("Alternative must be either 'two-sided', "
                         "'greater' or 'less'")

    if y is None:
        d = asarray(x)
        if d.ndim > 1:
            raise ValueError('Sample x must be one-dimensional.')
    else:
        x, y = map(asarray, (x, y))
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError('Samples x and y must be one-dimensional.')
        if len(x) != len(y):
            raise ValueError('The samples x and y must have the same length.')
        # Future enhancement: consider warning when elements of `d` appear to
        # be tied but are numerically distinct.
        d = x - y

    if len(d) == 0:
        NaN = _get_nan(d)
        res = WilcoxonResult(NaN, NaN)
        if method == 'approx':
            res.zstatistic = NaN
        return res

    if mode == "auto":
        if len(d) <= 50:
            mode = "exact"
        else:
            mode = "approx"

    n_zero = np.sum(d == 0)
    if n_zero > 0 and mode == "exact":
        mode = "approx"
        warnings.warn("Exact p-value calculation does not work if there are "
                      "zeros. Switching to normal approximation.",
                      stacklevel=2)

    if mode == "approx":
        if zero_method in ["wilcox", "pratt"]:
            if n_zero == len(d):
                raise ValueError("zero_method 'wilcox' and 'pratt' do not "
                                 "work if x - y is zero for all elements.")
        if zero_method == "wilcox":
            # Keep all non-zero differences
            d = compress(np.not_equal(d, 0), d)

    count = len(d)
    if count < 10 and mode == "approx":
        warnings.warn("Sample size too small for normal approximation.", stacklevel=2)

    r = _stats_py.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r)
    r_minus = np.sum((d < 0) * r)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    # return min for two-sided test, but r_plus for one-sided test
    # the literature is not consistent here
    # r_plus is more informative since r_plus + r_minus = count*(count+1)/2,
    # i.e. the sum of the ranks, so r_minus and the min can be inferred
    # (If alternative='pratt', r_plus + r_minus = count*(count+1)/2 - r_zero.)
    # [3] uses the r_plus for the one-sided test, keep min for two-sided test
    # to keep backwards compatibility
    if alternative == "two-sided":
        T = min(r_plus, r_minus)
    else:
        T = r_plus

    if mode == "approx":
        mn = count * (count + 1.) * 0.25
        se = count * (count + 1.) * (2. * count + 1.)

        if zero_method == "pratt":
            r = r[d != 0]
            # normal approximation needs to be adjusted, see Cureton (1967)
            mn -= n_zero * (n_zero + 1.) * 0.25
            se -= n_zero * (n_zero + 1.) * (2. * n_zero + 1.)

        replist, repnum = find_repeats(r)
        if repnum.size != 0:
            # Correction for repeated elements.
            se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

        se = sqrt(se / 24)

        # apply continuity correction if applicable
        d = 0
        if correction:
            if alternative == "two-sided":
                d = 0.5 * np.sign(T - mn)
            elif alternative == "less":
                d = -0.5
            else:
                d = 0.5

        # compute statistic and p-value using normal approximation
        z = (T - mn - d) / se
        if alternative == "two-sided":
            prob = 2. * distributions.norm.sf(abs(z))
        elif alternative == "greater":
            # large T = r_plus indicates x is greater than y; i.e.
            # accept alternative in that case and return small p-value (sf)
            prob = distributions.norm.sf(z)
        else:
            prob = distributions.norm.cdf(z)
    elif mode == "exact":
        # get pmf of the possible positive ranksums r_plus
        pmf = _get_wilcoxon_distr(count)
        # note: r_plus is int (ties not allowed), need int for slices below
        r_plus = int(r_plus)
        if alternative == "two-sided":
            if r_plus == (len(pmf) - 1) // 2:
                # r_plus is the center of the distribution.
                prob = 1.0
            else:
                p_less = np.sum(pmf[:r_plus + 1])
                p_greater = np.sum(pmf[r_plus:])
                prob = 2*min(p_greater, p_less)
        elif alternative == "greater":
            prob = np.sum(pmf[r_plus:])
        else:
            prob = np.sum(pmf[:r_plus + 1])
        prob = np.clip(prob, 0, 1)

    res = WilcoxonResult(T, prob)
    if method == 'approx':
        res.zstatistic = z
    return res, r_plus, r_minus