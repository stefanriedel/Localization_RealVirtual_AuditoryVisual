import numpy as np
from Utility.pairwiseTests import posthoc_wilcoxon, posthoc_ttest_ind
import matplotlib.pyplot as plt


POOL_METHODS = True
DYNAMIC = False

# Subjects that took both experiments (repeated) vs. just one (new)
subjects_repeated_exp1 = [1,2,3,8,9,13,12,15]
subjects_new_exp1 = [4,5,6,7,10,12,14,16]

subjects_repeated_exp2 = [13,3,12,5,4,1,11,2]
subjects_new_exp2 = [6,7,8,9,10,14,15,16]


idcs_repeated_exp1 = np.asarray(subjects_repeated_exp1) - 1
idcs_new_exp1 = np.asarray(subjects_new_exp1) - 1


idcs_repeated_exp2 = np.asarray(subjects_repeated_exp2) - 1
idcs_new_exp2 = np.asarray(subjects_new_exp2) - 1

if DYNAMIC:
    all_data = np.load('ErrorMetricData/Dynamic_Frontal_ElevationError.npy', allow_pickle=True).tolist()
    idcs_repeated = idcs_repeated_exp2
    idcs_new = idcs_new_exp2
else:
    all_data = np.load('ErrorMetricData/Static_Frontal_ElevationError.npy', allow_pickle=True).tolist()
    idcs_repeated = idcs_repeated_exp1
    idcs_new = idcs_new_exp1

if DYNAMIC:
    data_open_ears = all_data['DynamicOpenEars']
    data_open_hp = all_data['DynamicOpenHeadphones']
    data_ku100 = all_data['DynamicKU100HRTF']
    data_kemar = all_data['DynamicKEMARHRTF']
else:
    data_open_ears = all_data['StaticOpenEars']
    data_open_hp = all_data['StaticOpenHeadphones']
    data_ku100 = all_data['StaticKU100HRTF']
    data_indiv = all_data['StaticIndivHRTF']

if POOL_METHODS:
    if DYNAMIC:
        data = np.array([data_open_ears, data_open_hp, data_ku100, data_kemar])
    else:
        data = np.array([data_open_ears, data_open_hp, data_ku100, data_indiv])
    data_group_repeated = np.nanmean(data[:, idcs_repeated], axis=0)
    data_group_new = np.nanmean(data[:, idcs_new], axis=0)
else:
    data_group_repeated = data_ku100[idcs_repeated]
    data_group_new = data_ku100[idcs_new]

data = np.array([data_group_repeated, data_group_new])

pairs = [[0, 1]]

# pval_wilcoxon, effect_size = posthoc_wilcoxon(
#     data,
#     pairs,
#     alternative_h='two-sided',
#     p_adjust=None)
# print('Wilcoxon pval:' + str(round(pval_wilcoxon[0],3)))
 

pval_ttest = posthoc_ttest_ind(
    data,
    pairs,
    p_adjust=None)
print('t-test pval:' + str(round(pval_ttest[0],3)))


plt.figure()
plt.boxplot(data.T, labels=['Group 1 (both exp.)', 'Group 2 (one exp.)'])
plt.ylabel('Unsigned elevation error (°)')
if DYNAMIC:
    plt.title('Frontal Loudspeakers - Dynamic Experiment (Exp. 2)')
    savename = 'DYNAMIC_EXP2'
else:
    plt.title('Frontal Loudspeakers - Static Experiment (Exp. 1)')
    savename = 'STATIC_EXP1'

plt.ylim(0,30)

plt.text(1.25, 2, 't-test: p = ' + str(round(pval_ttest[0],3)), fontsize=12)

plt.savefig(fname='Figures/GroupAnalysis_' + savename + '.png', bbox_inches='tight', dpi=300)
plt.show()


print('done')