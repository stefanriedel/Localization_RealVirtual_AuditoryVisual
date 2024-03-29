import numpy as np
import scipy.stats as stats
# from cliffs_delta import cliffs_delta

from Utility.wilcoxon import stats_wilcoxon

def posthoc_wilcoxon(data, pairs_to_be_tested, method='auto', alternative_h='two-sided', p_adjust='BH', alpha=0.05, CL_ES='greater'):
    """ Perform pairwise Wilcoxon signed-rank tests.
    data (ndarray): condition x subject matrix
    pairs_to_be_tested: list of integers pairs for pairwise comparisons
    """
    pvals = np.zeros(len(pairs_to_be_tested))
    effect_sizes = np.zeros(len(pairs_to_be_tested))
    test_statistics = np.zeros(len(pairs_to_be_tested))
    z_statistics = np.zeros(len(pairs_to_be_tested)) 

    idx = 0
    for pair in pairs_to_be_tested:
        non_nan_idcs = np.logical_and(
            ~np.isnan(data[pair[0], :]), ~np.isnan(data[pair[1], :]))

        res, T_plus, T_minus = stats_wilcoxon(
            data[pair[0], non_nan_idcs], data[pair[1], non_nan_idcs],
            alternative=alternative_h, method=method)

        T = T_plus - T_minus

        if CL_ES == 'greater':
            order = data[pair[0], non_nan_idcs] > data[pair[1], non_nan_idcs]
        else:
            order = data[pair[1], non_nan_idcs] > data[pair[0], non_nan_idcs]
        common_language_effect_size = np.sum(order) / float(non_nan_idcs.size)

        pvals[idx] = res.pvalue
        if method == 'approx': # normal approximation, only meaningful for N >= 30
            z = res.zstatistic
            z_statistics[idx] = z
            effect_sizes[idx] = z / np.sqrt(non_nan_idcs.size)
        else:
            effect_sizes[idx] = common_language_effect_size

        test_statistics[idx] = T # signed-rank sum T = sum_i sgn(X_i) R_i 
        idx += 1

    if p_adjust == 'BH':
        # Sort p-values from smallest to largest
        sort_indices = np.argsort(pvals)
        corr_factor = pvals.size
        for i in range(pvals.size):
            pval_before_correction = pvals[sort_indices[i]]
            # Correction in style of Bonferroni-Holm
            pvals[sort_indices[i]] *= corr_factor

            if (pvals[sort_indices[i]] < pvals[sort_indices[i-1]]) and (i > 0) and pvals[sort_indices[i-1]] >= alpha:
                # EXIT: Avoids a p-value to be significant when the previous/smaller value
                # was insignificant after correction (meaning we should EXIT).
                # In this case, just apply more pessimistic correction by multiplying with number of tests (Bonferroni).
                pvals[sort_indices[i]] = pval_before_correction * (corr_factor + 1)

            # Clip p-values to 1.0
            if pvals[sort_indices[i]] > 1.0:
                pvals[sort_indices[i]] = 1.0

            # Update correction factor
            corr_factor -= 1
    if p_adjust == 'Bonferroni':
        pvals *= pvals.size


    return pvals, effect_sizes, test_statistics, z_statistics


def posthoc_ttest(data, pairs_to_be_tested, alternative_h, p_adjust='BH', alpha=0.05):
    """ Perform pairwise t test.
    data (ndarray): condition x subject matrix
    pairs_to_be_tested: list of integers pairs for pairwise comparisons
    """
    pvals = np.zeros(len(pairs_to_be_tested))
    idx = 0
    for pair in pairs_to_be_tested:
        non_nan_idcs = np.logical_and(
            ~np.isnan(data[pair[0], :]), ~np.isnan(data[pair[1], :]))

        t, pvals[idx] = stats.ttest_rel(
            data[pair[0], non_nan_idcs], data[pair[1], non_nan_idcs], alternative=alternative_h)
        idx += 1

    if p_adjust == 'BH':
        sort_indices = np.argsort(pvals)
        corr_factor = pvals.size
        for i in range(pvals.size):
            pval_before_correction = pvals[sort_indices[i]]
            # Correction in style of Bonferroni-Holm
            pvals[sort_indices[i]] *= corr_factor

            if (pvals[sort_indices[i]] < pvals[sort_indices[i-1]]) and (i > 0) and pvals[sort_indices[i-1]] >= alpha:
                # EXIT: Avoids a p-value to be significant when the previous/smaller value
                # was insignificant after correction (meaning we should EXIT).
                # In this case, just apply more pessimistic correction by multiplying with number of tests (Bonferroni).
                pvals[sort_indices[i]] = pval_before_correction * (corr_factor + 1)

            # Clip p-values to 1.0
            if pvals[sort_indices[i]] > 1.0:
                pvals[sort_indices[i]] = 1.0

            # Update correction factor
            corr_factor -= 1

    return pvals, t

def posthoc_ttest_ind(data, pairs_to_be_tested, p_adjust='BH'):
    """ Perform pairwise t test.
    data (ndarray): condition x subject matrix
    pairs_to_be_tested: list of integers pairs for pairwise comparisons
    """
    pvals = np.zeros(len(pairs_to_be_tested))
    idx = 0
    for pair in pairs_to_be_tested:
        non_nan_idcs = np.logical_and(
            ~np.isnan(data[pair[0], :]), ~np.isnan(data[pair[1], :]))

        w, pvals[idx] = stats.ttest_ind(
            data[pair[0], non_nan_idcs], data[pair[1], non_nan_idcs])
        idx += 1

    if p_adjust == 'BH':
        sort_indices = np.argsort(pvals)
        corr_factor = pvals.size
        for i in range(pvals.size):
            pvals[sort_indices[i]] *= corr_factor

            if (pvals[sort_indices[i]] < pvals[sort_indices[i-1]]) and (i > 0):
                # EXIT: If a p-value is smaller than the previous after the correction,
                # meaning a change in the order due to correction, clip it to the corrected previous value.
                # This avoids a p-value to be significant when the previous/smaller value
                # was insignificant after correction. This corresponds to an EXIT strategy.
                pvals[sort_indices[i]] = pvals[sort_indices[i-1]]

            # Clip p-values to 1.0
            if pvals[sort_indices[i]] > 1.0:
                pvals[sort_indices[i]] = 1.0

            # Update correction factor
            corr_factor -= 1

    return pvals

