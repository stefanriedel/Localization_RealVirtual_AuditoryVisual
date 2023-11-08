import numpy as np
import scipy.stats as stats
# from cliffs_delta import cliffs_delta


def posthoc_wilcoxon(data, pairs_to_be_tested, alternative_h='two-sided', p_adjust='BH'):
    """ Perform pairwise Wilcoxon signed-rank tests.
    data (ndarray): condition x subject matrix
    pairs_to_be_tested: list of integers pairs for pairwise comparisons
    """
    pvals = np.zeros(len(pairs_to_be_tested))
    effect_sizes = np.zeros(len(pairs_to_be_tested))

    idx = 0
    for pair in pairs_to_be_tested:
        non_nan_idcs = np.logical_and(
            ~np.isnan(data[pair[0], :]), ~np.isnan(data[pair[1], :]))

        res = stats.wilcoxon(
            data[pair[0], non_nan_idcs], data[pair[1], non_nan_idcs],
            alternative=alternative_h, method='approx')

        total_rank_sum = 0
        for i in range(1, non_nan_idcs.size+1):
            total_rank_sum += i
        pvals[idx] = res.pvalue
        z = res.zstatistic
        effect_sizes[idx] = z / np.sqrt(non_nan_idcs.size)
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

    return pvals, effect_sizes


def posthoc_ttest(data, pairs_to_be_tested, p_adjust='BH'):
    """ Perform pairwise t test.
    data (ndarray): condition x subject matrix
    pairs_to_be_tested: list of integers pairs for pairwise comparisons
    """
    pvals = np.zeros(len(pairs_to_be_tested))
    idx = 0
    for pair in pairs_to_be_tested:
        non_nan_idcs = np.logical_and(
            ~np.isnan(data[pair[0], :]), ~np.isnan(data[pair[1], :]))

        w, pvals[idx] = stats.ttest_rel(
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


# def posthoc_cliffs_delta(data, pairs_to_be_tested):
#     cliffs_d = np.zeros(len(pairs_to_be_tested))
#     idx = 0
#     for pair in pairs_to_be_tested:
#         non_nan_idcs = np.logical_and(
#             ~np.isnan(data[pair[0], :]), ~np.isnan(data[pair[1], :]))

#         cliffs_d[idx], res = cliffs_delta(
#             data[pair[1], non_nan_idcs].tolist(), data[pair[0], non_nan_idcs].tolist())
#         idx += 1

#     return cliffs_d
