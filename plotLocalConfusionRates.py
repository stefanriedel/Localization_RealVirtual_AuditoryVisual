from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ErrorMetricData')
fig_dir = pjoin(root_dir, 'Figures')

# LOAD ALL PRECOMPUTED DATA
lcr_static_ele = np.load(file=pjoin(data_dir, 'LocalConfusionDataElevationStatic.npy'), allow_pickle=True).tolist()
lcr_static_azi = np.load(file=pjoin(data_dir, 'LocalConfusionDataAzimuthStatic.npy'), allow_pickle=True).tolist()
slope_static_ele = np.load(file=pjoin(data_dir, 'SlopeDataElevationStatic.npy'), allow_pickle=True).tolist()
slope_static_azi = np.load(file=pjoin(data_dir, 'SlopeDataAzimuthStatic.npy'), allow_pickle=True).tolist()

lcr_dynamic_ele = np.load(file=pjoin(data_dir, 'LocalConfusionDataElevationDynamic.npy'), allow_pickle=True).tolist()
lcr_dynamic_azi = np.load(file=pjoin(data_dir, 'LocalConfusionDataAzimuthDynamic.npy'), allow_pickle=True).tolist()
slope_dynamic_ele = np.load(file=pjoin(data_dir, 'SlopeDataElevationDynamic.npy'), allow_pickle=True).tolist()
slope_dynamic_azi = np.load(file=pjoin(data_dir, 'SlopeDataAzimuthDynamic.npy'), allow_pickle=True).tolist()

class median_iqr:
    def __init__(self):
        self.low = 0
        self.high = 0

def plotLocalConfusionRatesConstantSampleSize(lcr_data, conditions, directions, xlabel, ylabel, savename):
    confusion_rates = np.zeros((4,16))

    for condition, condition_idx in zip(conditions, range(len(conditions))):
        for subj in range(16):
            if condition == 'StaticOpenEars' or condition == 'DynamicOpenEars':
                confusions = np.asarray(lcr_data[condition]['Confusions'])[directions,:][:, subj]
            else:
                confusions = np.asarray(lcr_data[condition]['Confusions'])[directions,:][:, [subj, subj + 16]]
                confusions = np.nanmean(confusions, axis=1)

            n = np.sum(~np.isnan(confusions)) # Number of local data points
            k = np.nansum(confusions) # Number of confusion in local data points
            confusion_rates[condition_idx, subj] = k / n

    plt.figure(figsize=(2.5,2.5))
    plt.grid(axis='y')
    #plt.boxplot(confusion_rates.T * 100.0, labels = xlabel)
    sns.violinplot(confusion_rates.T * 100.0)
    plt.xticks([0,1,2,3], xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0,100.0])
    plt.yticks(ticks=np.arange(0, 110, 10))
    plt.tight_layout()
    plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
    plt.show(block=True)

    return

def plotLocalConfusionRatesConstantSampleSizeDensity(lcr_data, conditions, density_directions, xlabel, ylabel, savename):
    confusion_rates = np.zeros((8,16))

    condition_idx = 0
    for condition in conditions:
        for directions in density_directions:
            for subj in range(16):
                if condition == 'StaticOpenEars' or condition == 'DynamicOpenEars':
                    confusions = np.asarray(lcr_data[condition]['Confusions'])[directions,:][:, subj]
                else:
                    confusions = np.asarray(lcr_data[condition]['Confusions'])[directions,:][:, [subj, subj + 16]]
                    confusions = np.nanmean(confusions, axis=1)

                n = np.sum(~np.isnan(confusions)) # Number of local data points
                k = np.nansum(confusions) # Number of confusion in local data points
                confusion_rates[condition_idx, subj] = k / n

            condition_idx += 1

    median_iqr_density = median_iqr()
    median_iqr_density.low = np.quantile(confusion_rates*100.0, q=0.25, axis=1)
    median_iqr_density.high = np.quantile(confusion_rates*100.0, q=0.75, axis=1)

    plt.figure(figsize=(2.5,2.5))
    plt.grid(axis='y')
    #plt.boxplot(confusion_rates.T * 100.0, labels = xlabel)
    offs = 0.25
    #plt.plot([1-offs,1+offs, 2-offs,2+offs, 3-offs,3+offs, 4-offs, 4+offs], np.median(confusion_rates.T * 100.0, axis=0), ls='', marker='s', markerfacecolor='white', markeredgecolor='k')
    medians = np.median(confusion_rates.T * 100.0, axis=0)
    #plt.errorbar([1-offs,1+offs, 2-offs,2+offs, 3-offs,3+offs, 4-offs, 4+offs], 
    #             medians, yerr=[medians-median_iqr_density.low, median_iqr_density.high-medians], 
    #             ls='', marker='s', markerfacecolor='white', markeredgecolor='k', capsize=1.5, color='k')
    
    h_idx = [0,2,4,6]
    plt.errorbar([1-offs, 2-offs, 3-offs, 4-offs], 
                 medians[h_idx], yerr=[medians[h_idx]-median_iqr_density.low[h_idx], median_iqr_density.high[h_idx]-medians[h_idx]], 
                 ls='', marker='s', markerfacecolor='white', markeredgecolor='tab:blue', capsize=2.5, color='tab:blue', label='Dense (15° vert. spacing)')
    l_idx = [1,3,5,7]
    plt.errorbar([1+offs, 2+offs, 3+offs, 4+offs], 
                 medians[l_idx], yerr=[medians[l_idx]-median_iqr_density.low[l_idx], median_iqr_density.high[l_idx]-medians[l_idx]], 
                 ls='', marker='s', markerfacecolor='white', markeredgecolor='k', capsize=2.5, color='k', label='Sparse (30° vert. spacing)')
    
    plt.xticks([1,2,3,4], xlabel)
    plt.plot([1.5,1.5], [0,100], color='gray')
    plt.plot([2.5,2.5], [0,100], color='gray')
    plt.plot([3.5,3.5], [0,100], color='gray')
    plt.plot([4.5,4.5], [0,100], color='gray')

    plt.ylabel(ylabel)
    plt.ylim([0,100.0])
    plt.yticks(ticks=np.arange(0, 110, 10))
    plt.legend(framealpha=1.0)
    plt.tight_layout()
    plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
    plt.show(block=True)
    
    return

frontal_directions = [23, 0, 24]
#directions = frontal_directions#[*range(20)] + [*range(21, 25)]

conditions = ['StaticOpenEars', 'StaticOpenHeadphones', 'StaticIndivHRTF', 'StaticKU100HRTF']
#conditions = ['StaticKU100HRTF']


if 1:
    # STATIC
    directions = [*range(20)] 
    xlabel = ['REF', 'OH', 'Indiv. ' , ' KU100']
    ylabel = 'Horizontal LCR (%)' 
    savename = 'StaticHorizontalLCR.eps'
    plotLocalConfusionRatesConstantSampleSize(lcr_static_azi, conditions , directions, xlabel, ylabel, savename)

    #directions = [24, 0, 23, 8] + [1, 9] + [7, 22, 15] + [2, 10, 17] + [6, 21, 14, 19]
    directions = [*range(20)] + [*range(21, 25)]
    xlabel = ['REF', 'OH', 'Indiv. ' , ' KU100']
    ylabel = 'Vertical LCR (%)' 
    savename = 'StaticVerticalLCR.eps'
    plotLocalConfusionRatesConstantSampleSize(lcr_static_ele, conditions , directions, xlabel, ylabel, savename)
if 0:
    conditions = ['DynamicOpenEars', 'DynamicOpenHeadphones', 'DynamicKEMARHRTF', 'DynamicKU100HRTF']

    # DYNAMIC
    directions = [*range(20)] 
    xlabel = ['REF', 'OH', 'KEMAR' , 'KU100']
    ylabel = 'Horizontal LCR (%)' 
    savename = 'DynamicHorizontalLCR.eps'
    plotLocalConfusionRatesConstantSampleSize(lcr_dynamic_azi, conditions , directions, xlabel, ylabel, savename)

    #directions = [24, 0, 23, 8] + [1, 9] + [7, 22, 15] + [2, 10, 17] + [6, 21, 14, 19]
    directions = [*range(20)] + [*range(21, 25)]
    xlabel = ['REF', 'OH', 'KEMAR' , ' KU100']
    ylabel = 'Vertical LCR (%)' 
    savename = 'DynamicVerticalLCR.eps'
    plotLocalConfusionRatesConstantSampleSize(lcr_dynamic_ele, conditions , directions, xlabel, ylabel, savename)




if 1:
    # Frontal
    directions = [0, 23, 24, 1, 7]
    #directions = [*range(20)] + [*range(21, 25)]
    xlabel = ['REF', 'OH', 'Indiv. ' , ' KU100']
    ylabel = 'Vertical LCR (%)' 
    savename = 'StaticVerticalLCRFrontal.eps'
    plotLocalConfusionRatesConstantSampleSize(lcr_static_ele, conditions , directions, xlabel, ylabel, savename)
if 0:
    # Dense
    directions = [7, 22, 15, 6, 21, 14] #+ [0, 23, 24, 8]
    #directions = [*range(20)] + [*range(21, 25)]
    xlabel = ['REF', 'OH', 'Indiv. ' , ' KU100']
    ylabel = 'Vertical LCR (%)' 
    savename = 'StaticVerticalLCRDense.eps'
    plotLocalConfusionRatesConstantSampleSize(lcr_static_ele, conditions , directions, xlabel, ylabel, savename)

    # Sparse
    directions = [1, 9, 2, 10]
    #directions = [*range(20)] + [*range(21, 25)]
    xlabel = ['REF', 'OH', 'Indiv. ' , ' KU100']
    ylabel = 'Vertical LCR (%)' 
    savename = 'StaticVerticalLCRSparse.eps'
    plotLocalConfusionRatesConstantSampleSize(lcr_static_ele, conditions , directions, xlabel, ylabel, savename)


if 1:
    xlabel = ['REF', 'OH', 'Indiv. ' , ' KU100']
    ylabel = 'Vertical LCR (%)' 
    savename = 'StaticVerticalLCR_DenseVsSparse.eps'
    density_directions = [[7, 22, 15, 6, 21, 14], [1, 9, 2, 10]]#[[7, 15, 6, 14], [1, 9, 2, 10]]#[[7, 22, 15, 6, 21, 14], [1, 9, 2, 10]]
    plotLocalConfusionRatesConstantSampleSizeDensity(lcr_static_ele, conditions , density_directions, xlabel, ylabel, savename)


