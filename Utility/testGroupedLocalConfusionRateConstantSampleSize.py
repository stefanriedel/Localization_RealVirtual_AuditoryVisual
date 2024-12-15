import numpy as np
import scipy.stats as stats
from .wilcoxon import stats_wilcoxon
from .mannwhitneyu import stats_mannwhitneyu 
from cliffs_delta import cliffs_delta
from CLES import CLES_2paired, CLES_2independent

def testGroupedLocalConfusionRateConstantSampleSize(first_condition_data, second_condition_data, condition_pair, direction_pair, SUBJ_IDCS=[np.arange(16), np.arange(16)], PAIRED_SAMPLES=True, NONPARAM=True):
    # Only test over equal numbers of directions
    assert(len(direction_pair[0]) == len(direction_pair[1]))

    confusion_rate_first = np.zeros(len(SUBJ_IDCS[0]))
    confusion_rate_second = np.zeros(len(SUBJ_IDCS[0]))

    for subj_first, subj_second, i in zip(SUBJ_IDCS[0], SUBJ_IDCS[1], range(len(SUBJ_IDCS[0]))):
        if condition_pair[0] == 'StaticOpenEars' or condition_pair[0] == 'DynamicOpenEars':
            first_confusions = np.asarray(first_condition_data[condition_pair[0]]['Confusions'])[direction_pair[0],:][:, subj_first]
        else:
            first_confusions = np.asarray(first_condition_data[condition_pair[0]]['Confusions'])[direction_pair[0],:][:, [subj_first, subj_first + 16]]
            first_confusions = np.nanmean(first_confusions, axis=1)

        if condition_pair[1] == 'StaticOpenEars' or condition_pair[1] == 'DynamicOpenEars':
            second_confusions = np.asarray(second_condition_data[condition_pair[1]]['Confusions'])[direction_pair[1],:][:, subj_second]
        else:
            second_confusions = np.asarray(second_condition_data[condition_pair[1]]['Confusions'])[direction_pair[1],:][:, [subj_second, subj_second + 16]]
            second_confusions = np.nanmean(second_confusions, axis=1)

        n_first = np.sum(~np.isnan(first_confusions)) # Number of local data points
        k_first = np.nansum(first_confusions) # Number of confusion in local data points
        confusion_rate_first[i] = k_first / n_first

        n_second = np.sum(~np.isnan(second_confusions)) # Number of local data points
        k_second = np.nansum(second_confusions) # Number of confusion in local data points
        confusion_rate_second[i] = k_second / n_second

    if PAIRED_SAMPLES:
        if NONPARAM:
            res, T_plus, T_minus = stats_wilcoxon(confusion_rate_first, confusion_rate_second, alternative='two-sided', method='approx')
            effect_size = res.effect_size # effect size calculated from z value in approx method (requested for revision of paper)
            cliffs_d = cliffs_delta(confusion_rate_first, confusion_rate_second)
            cles = CLES_2paired(confusion_rate_first, confusion_rate_second)
            res, T_plus, T_minus = stats_wilcoxon(confusion_rate_first, confusion_rate_second, alternative='two-sided', method='auto')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 'T(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(T_plus - T_minus), 'r = ' + str(effect_size), 'cles = ' + str(cles))
        else:
            res = stats.ttest_rel(confusion_rate_first, confusion_rate_second, alternative='two-sided')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 't(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]))
    else:
        if NONPARAM:
            res, effect_size = stats_mannwhitneyu(confusion_rate_first, confusion_rate_second, alternative='two-sided')
            cles = CLES_2independent(confusion_rate_first, confusion_rate_second)
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 'U1(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]), 'r = ' + str(effect_size), 'cles = ' + str(cles))
        else:
            res = stats.ttest_ind(confusion_rate_first, confusion_rate_second, alternative='two-sided')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 't(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]))

    return res[1]