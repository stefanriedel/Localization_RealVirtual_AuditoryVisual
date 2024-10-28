from Utility.localizationEvaluationUtility import testGroupedLocalConfusionRate, testGroupedSlopeData, testGroupedLocalConfusionRateConstantSampleSize, testGroupedSlopeDataConstantSampleSize
from os.path import dirname, join as pjoin
import numpy as np


def bonferroniHolm(pvals):
    pvals = np.asarray(pvals)

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


# SET DIMENSION YOU WANT TO TEST
AZIMUTH = True
ELEVATION = not AZIMUTH

NONPARAM = True

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ErrorMetricData')

# LOAD ALL PRECOMPUTED DATA
lcr_static_ele = np.load(file=pjoin(data_dir, 'LocalConfusionDataElevationStatic.npy'), allow_pickle=True).tolist()
lcr_static_azi = np.load(file=pjoin(data_dir, 'LocalConfusionDataAzimuthStatic.npy'), allow_pickle=True).tolist()
slope_static_ele = np.load(file=pjoin(data_dir, 'SlopeDataElevationStatic.npy'), allow_pickle=True).tolist()
slope_static_azi = np.load(file=pjoin(data_dir, 'SlopeDataAzimuthStatic.npy'), allow_pickle=True).tolist()

lcr_dynamic_ele = np.load(file=pjoin(data_dir, 'LocalConfusionDataElevationDynamic.npy'), allow_pickle=True).tolist()
lcr_dynamic_azi = np.load(file=pjoin(data_dir, 'LocalConfusionDataAzimuthDynamic.npy'), allow_pickle=True).tolist()
slope_dynamic_ele = np.load(file=pjoin(data_dir, 'SlopeDataElevationDynamic.npy'), allow_pickle=True).tolist()
slope_dynamic_azi = np.load(file=pjoin(data_dir, 'SlopeDataAzimuthDynamic.npy'), allow_pickle=True).tolist()

# TESTS WITHIN STATIC CONDITIONS
condition_pairs = [['StaticOpenEars', 'StaticOpenHeadphones'], ['StaticOpenEars', 'StaticIndivHRTF'], ['StaticOpenEars', 'StaticKU100HRTF'], ['StaticOpenHeadphones', 'StaticIndivHRTF'], ['StaticOpenHeadphones', 'StaticKU100HRTF'], ['StaticIndivHRTF', 'StaticKU100HRTF']]
if AZIMUTH:
    # Static LCR Azimuth Tests
    directions = [*range(20)] # All directions of the three height layers
    directions = [directions, directions]

    pvals = []
    print('Static Azimuth LCR Tests: ')
    for condition_pair in condition_pairs:
        #testGroupedLocalConfusionRate(lcr_static_azi, lcr_static_azi, condition_pair, directions)
        p = testGroupedLocalConfusionRateConstantSampleSize(lcr_static_azi, lcr_static_azi, condition_pair, directions, NONPARAM=NONPARAM)
        pvals.append(p)
    pvals = bonferroniHolm(pvals)

    print('BH-corrected pvals: \n')
    for condition_pair, i in zip(condition_pairs, range(len(condition_pairs))):
        print(str(condition_pair) + ' p = ' + str(pvals[i]))
    print('')

    # Static Slope Azimuth Tests
    #planes = ['0DEG+30DEG+60DEG']
    planes = ['0DEG', '30DEG', '60DEG']
    print('Static Azimuth Slope Tests: ')
    for condition_pair in condition_pairs:
        testGroupedSlopeDataConstantSampleSize(slope_static_azi, slope_static_azi, condition_pair, planes, NONPARAM=NONPARAM)
    print('')

if ELEVATION:
    # Static LCR Elevation Tests
    print('Static Elevation LCR Test: ')
    directions = [*range(20)] + [*range(21, 25)] # All directions except the zenith
    directions = [directions, directions]
    for condition_pair in condition_pairs:
        #testGroupedLocalConfusionRate(lcr_static_ele, lcr_static_ele, condition_pair, directions)
        testGroupedLocalConfusionRateConstantSampleSize(lcr_static_ele, lcr_static_ele, condition_pair, directions, NONPARAM=NONPARAM)

    print('')

    # Static Slope Elevation Tests
    planes = ['0DEG', '30DEG', '-30DEG', '90DEG', '-90DEG', '150DEG', '-150DEG', '180DEG']
    print('Static Elevation Slope Tests: ')
    for condition_pair in condition_pairs:
        testGroupedSlopeDataConstantSampleSize(slope_static_ele, slope_static_ele, condition_pair, planes, NONPARAM=NONPARAM)
    print('')

# TESTS WITHIN DYNAMIC CONDITIONS
condition_pairs = [['DynamicOpenEars', 'DynamicOpenHeadphones'], ['DynamicOpenEars', 'DynamicKEMARHRTF'], ['DynamicOpenEars', 'DynamicKU100HRTF'], 
                    ['DynamicOpenHeadphones', 'DynamicKEMARHRTF'], ['DynamicOpenHeadphones', 'DynamicKU100HRTF'] , ['DynamicKEMARHRTF', 'DynamicKU100HRTF']]
if AZIMUTH:
    # Dynamic LCR Azimuth Tests
    directions = [*range(20)] # All directions of the three height layers
    directions = [directions, directions]
    print('Dynamic Azimuth LCR Tests: ')
    for condition_pair in condition_pairs:
        testGroupedLocalConfusionRateConstantSampleSize(lcr_dynamic_azi, lcr_dynamic_azi, condition_pair, directions, NONPARAM=NONPARAM)
    print('')

    # Dynamic Slope Azimuth Tests
    #planes = ['0DEG+30DEG+60DEG']
    planes = ['0DEG', '30DEG', '60DEG']
    print('Dynamic Azimuth Slope Tests: ')
    for condition_pair in condition_pairs:
        testGroupedSlopeDataConstantSampleSize(slope_dynamic_azi, slope_dynamic_azi, condition_pair, planes, NONPARAM=NONPARAM)
    print('')
if ELEVATION:
    # Dynamic LCR Elevation Tests
    directions = [*range(20)] + [*range(21, 25)] # All directions except the zenith
    directions = [directions, directions]
    print('Dynamic Elevation LCR Tests: ')
    for condition_pair in condition_pairs:
        testGroupedLocalConfusionRateConstantSampleSize(lcr_dynamic_ele, lcr_dynamic_ele, condition_pair, directions, NONPARAM=NONPARAM)
    print('')

    # Dynamic Slope Elevation Tests
    planes = ['0DEG', '30DEG', '-30DEG', '90DEG', '-90DEG', '150DEG', '-150DEG', '180DEG']
    print('Dynamic Elevation Slope Tests: ')
    for condition_pair in condition_pairs:
        testGroupedSlopeDataConstantSampleSize(slope_dynamic_ele, slope_dynamic_ele, condition_pair, planes, NONPARAM=NONPARAM)
    print('')


# TESTS STATIC VS. DYNAMIC CONDITIONS
condition_pairs = [['StaticKU100HRTF', 'DynamicKU100HRTF'], [ 'StaticIndivHRTF', 'DynamicKU100HRTF'], ['StaticOpenHeadphones', 'DynamicOpenHeadphones'], ['StaticOpenEars', 'DynamicOpenEars']]

# Compare only participants that took part in both experiments
#subjects_repeated_exp1 = np.array([1,2,3,8,9,13,12,15])
#subjects_repeated_exp2 = np.array([13,3,12,5,4,1,11,2])

#subject_idcs = [subjects_repeated_exp1, subjects_repeated_exp2]

if AZIMUTH:
    # StaticVSDynamic LCR Azimuth Tests
    directions = [*range(20)] # All directions of the three height layers
    directions = [directions, directions]
    print('StaticVSDynamic Azimuth LCR Tests: ')
    for condition_pair in condition_pairs:
        #testGroupedLocalConfusionRate(lcr_static_azi, lcr_dynamic_azi, condition_pair, directions, SUBJ_IDCS=subject_idcs)
        testGroupedLocalConfusionRateConstantSampleSize(lcr_static_azi, lcr_dynamic_azi, condition_pair, directions, PAIRED_SAMPLES=False, NONPARAM=NONPARAM)
    print('')

    # StaticVSDynamic Slope Azimuth Tests
    #planes = ['0DEG+30DEG+60DEG']
    planes = ['0DEG', '30DEG', '60DEG']
    print('StaticVSDynamic Azimuth Slope Tests: ')
    for condition_pair in condition_pairs:
        testGroupedSlopeDataConstantSampleSize(slope_static_azi, slope_dynamic_azi, condition_pair, planes, PAIRED_SAMPLES=False, NONPARAM=NONPARAM)
    print('')
if ELEVATION:
    # StaticVSDynamic LCR Elevation Tests
    directions = [*range(20)] + [*range(21, 25)] # All directions except the zenith
    directions = [directions, directions]
    print('StaticVSDynamic Elevation LCR Tests: ')
    for condition_pair in condition_pairs:
        #testGroupedLocalConfusionRate(lcr_static_ele, lcr_dynamic_ele, condition_pair, directions, SUBJ_IDCS=subject_idcs)
        testGroupedLocalConfusionRateConstantSampleSize(lcr_static_ele, lcr_dynamic_ele, condition_pair, directions, PAIRED_SAMPLES=False, NONPARAM=NONPARAM)
    print('')

    # StaticVSDynamic Slope Elevation Tests
    planes = ['0DEG', '30DEG', '-30DEG', '90DEG', '-90DEG', '150DEG', '-150DEG', '180DEG']
    print('StaticVSDynamic Elevation Slope Tests: ')
    for condition_pair in condition_pairs:
        testGroupedSlopeDataConstantSampleSize(slope_static_ele, slope_dynamic_ele, condition_pair, planes, PAIRED_SAMPLES=False, NONPARAM=NONPARAM)
    print('')

# TESTS ON VISUAL ANCHOR DENSITY: STATIC
condition_pairs = [['StaticKU100HRTF', 'StaticKU100HRTF'], [ 'StaticIndivHRTF', 'StaticIndivHRTF'], ['StaticOpenHeadphones', 'StaticOpenHeadphones'], ['StaticOpenEars', 'StaticOpenEars']]
# Visual Anchor Density LCR Elevation Tests
directions = [[1,9,2,10], [7,15,6,14]]
#directions = [[1,9], [7,15]]

print('Visual Anchor Density LCR Elevation Tests: ')
for condition_pair in condition_pairs:
    testGroupedLocalConfusionRateConstantSampleSize(lcr_static_ele, lcr_static_ele, condition_pair, directions, NONPARAM=NONPARAM)
print('')

# TESTS ON VISUAL ANCHOR DENSITY: DYNAMIC
condition_pairs = [['DynamicKU100HRTF', 'DynamicKU100HRTF'], [ 'DynamicKEMARHRTF', 'DynamicKEMARHRTF'], ['DynamicOpenHeadphones', 'DynamicOpenHeadphones'], ['DynamicOpenEars', 'DynamicOpenEars']]
# Visual Anchor Density LCR Elevation Tests
directions = [[1,9,2,10], [7,15,6,14]]
#directions = [[1,9], [7,15]]

print('Visual Anchor Density LCR Elevation Tests: ')
for condition_pair in condition_pairs:
    testGroupedLocalConfusionRateConstantSampleSize(lcr_dynamic_ele, lcr_dynamic_ele, condition_pair, directions, NONPARAM=NONPARAM)
print('')


 