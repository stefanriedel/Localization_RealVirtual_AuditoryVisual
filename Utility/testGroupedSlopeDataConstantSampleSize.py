import numpy as np
import scipy.stats as stats

from .wilcoxon import stats_wilcoxon
from .mannwhitneyu import stats_mannwhitneyu 

def testGroupedSlopeDataConstantSampleSize(first_condition_data, second_condition_data, condition_pair, planes, SUBJ_IDCS=[np.arange(16), np.arange(16)], PAIRED_SAMPLES=True, NONPARAM=True):
    g_first = np.zeros((len(SUBJ_IDCS[0]), len(planes)))
    g_second = np.zeros((len(SUBJ_IDCS[1]), len(planes)))

    for plane, i in zip(planes, range(len(planes))):
        first_data = first_condition_data[condition_pair[0]][plane]
        second_data = second_condition_data[condition_pair[1]][plane]

        first_data = first_data[SUBJ_IDCS[0]]
        second_data = second_data[SUBJ_IDCS[1]]

        g_first[:, i] = first_data
        g_second[:, i] = second_data

    g_first = np.nanmean(g_first, axis=1)
    g_second = np.nanmean(g_second, axis=1)

    non_nan_idcs_first = ~np.isnan(g_first)
    non_nan_idcs_second = ~np.isnan(g_second)
    non_nan_idcs = np.logical_and(non_nan_idcs_first, non_nan_idcs_second)

    g_first = g_first[non_nan_idcs]
    g_second = g_second[non_nan_idcs]

    if PAIRED_SAMPLES:
        if NONPARAM:
            res, T_plus, T_minus = stats_wilcoxon(g_first, g_second, alternative='two-sided')
            print('Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.median(g_first), 2)) + ' vs. ' + str(np.round(np.median(g_second), 2)),
                'pvalue: ' + str(res[1]),
                'T(' + str(g_first.size) + ') = ' + str(T_plus - T_minus))
        else:
            res = stats.ttest_rel(g_first, g_second, alternative='two-sided')
            print('Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.mean(g_first), 2)) + ' vs. ' + str(np.round(np.mean(g_second), 2)),
                'pvalue: ' + str(res[1]),
                't(' + str(g_first.size) + ') = ' + str(res[0]))
    else:
        if NONPARAM:
            res = stats.mannwhitneyu(g_first, g_second, alternative='two-sided')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.median(g_first), 2)) + ' vs. ' + str(np.round(np.median(g_second), 2)),
                'pvalue: ' + str(res[1]), 'U1(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]))
        else:
            res = stats.ttest_ind(g_first, g_second, alternative='two-sided')
            print('Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.mean(g_first), 2)) + ' vs. ' + str(np.round(np.mean(g_second), 2)),
                'pvalue: ' + str(res[1]),
                't(' + str(g_first.size) + ') = ' + str(res[0]))
    return  


