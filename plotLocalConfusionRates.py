from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt

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
    plt.boxplot(confusion_rates.T * 100.0, labels = xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0,100.0])
    plt.yticks(ticks=np.arange(0, 110, 10))
    plt.tight_layout()
    plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
    plt.show(block=True)

    return


frontal_directions = [23, 0, 24]
#directions = frontal_directions#[*range(20)] + [*range(21, 25)]

conditions = ['StaticOpenEars', 'StaticOpenHeadphones', 'StaticIndivHRTF', 'StaticKU100HRTF']
#conditions = ['StaticKU100HRTF']


if 0:
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

