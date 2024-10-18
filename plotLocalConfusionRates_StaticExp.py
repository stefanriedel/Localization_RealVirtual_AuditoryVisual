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

ylabel_textsize = 12

class median_iqr:
    def __init__(self):
        self.low = 0
        self.high = 0

def getConfusionRatesConstantSampleSize(lcr_data, conditions, directions):
    confusion_rates = np.zeros((len(conditions), 16))

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

    return confusion_rates



conditions = ['StaticOpenEars', 'StaticOpenHeadphones', 'StaticIndivHRTF', 'StaticKU100HRTF']
xlabel = ['Op.Ear ', ' Op.Hp.', 'Indiv.' , 'KU100']


# STATIC HORIZONTAL: GLOBAL
directions = [*range(20)] 
ylabel = 'Horizontal LCR (%)' 
savename = 'StaticHorizontalLCR.eps'

confusion_rates = getConfusionRatesConstantSampleSize(lcr_static_azi, conditions, directions)

plt.figure(figsize=(3,3))
plt.grid(axis='y')
ax = sns.violinplot(confusion_rates.T * 100.0, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'lightgreen', 'lightcoral'], inner_kws=dict(whis_width=2, color="black")) 
#plt.setp(ax.collections, alpha=.1)
plt.xticks([0,1,2,3], xlabel)

off = 0.075
if 0:
    # OpEar vs. OpHp
    plt.plot([0+off,0+off,1-off,1-off], [38,40,40,38], color='k')
    plt.text(x=0.5-0.075*3, y=40, s='***')

    # OpHp vs. Indiv.
    #plt.plot([1+off,1+off,2-off,2-off], [48,50,50,48], color='k')
    #plt.text(x=1.5-0.075*2, y=51, s='**')

    # Indiv. vs. KU100
    plt.plot([2+off,2+off,3-off,3-off], [38,40,40,38], color='k')
    plt.text(x=2.5-0.075*2, y=41, s='ns')

    # OpHp vs. KU100
    #plt.plot([1+off*2,1+off*2,3-off*2,3-off*2], [38,40,40,38], color='k')
    #plt.text(x=2-0.075*2, y=41, s='ns')

# Alternative: Connect all pariwise significant tests (p < 0.05) after Bonferroni correction
if 1:
    o = -6
    # OpEar vs. Rest
    plt.plot([0+off, 1-off], [84+o,84+o], color='k')
    plt.plot([0+off, 3-off], [86+o,86+o], color='k')
    # OpHp vs. Indiv. 
    plt.plot([1+off,2-off], [82+o,82+o], color='k')


plt.ylabel(ylabel, fontsize=ylabel_textsize)
plt.ylim([0,100.0])
plt.yticks(ticks=np.arange(0, 110, 10))
plt.tight_layout()
plt.title('Static')

plt.text(x=-0.1,y=-17,s='--Real--')
plt.text(x=1.9,y=-17,s='--Virtual--')

plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
plt.show(block=True)


# STATIC VERTICAL: GLOBAL
directions = [*range(20)] + [*range(21, 25)]
ylabel = 'Vertical LCR (%)' 
savename = 'StaticVerticalLCR.eps'

confusion_rates = getConfusionRatesConstantSampleSize(lcr_static_ele, conditions, directions)

plt.figure(figsize=(3,3))
plt.grid(axis='y')
sns.violinplot(confusion_rates.T * 100.0, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'lightgreen', 'lightcoral'], inner_kws=dict(whis_width=2, color="black"))
plt.xticks([0,1,2,3], xlabel)

off = 0.075
if 0:
    # OpEar vs. OpHp
    plt.plot([0+off,0+off,1-off,1-off], [88,90,90,88], color='k')
    plt.text(x=0.5-0.075*3, y=90, s='***')

    # OpHp vs. Indiv.
    plt.plot([1+off,1+off,2-off,2-off], [88,90,90,88], color='k')
    plt.text(x=1.5-0.075*2, y=91, s='ns')

    # Indiv. vs. KU100
    plt.plot([2+off,2+off,3-off,3-off], [88,90,90,88], color='k')
    plt.text(x=2.5-0.075*2, y=90, s='**')

    # OpHp vs. KU100
    plt.plot([1+off*2,1+off*2,3-off*2,3-off*2], [78,80,80,78], color='k')
    plt.text(x=2-0.075, y=80, s='*')

# Alternative: Connect all pariwise significant tests (p < 0.05) after Bonferroni correction

if 1:
    o = -8
    # OpEar vs. Rest
    plt.plot([0+off, 1-off], [84+o,84+o], color='k')
    plt.plot([0+off, 2-off], [86+o,86+o], color='k')
    plt.plot([0+off, 3-off], [88+o,88+o], color='k')
    # Indiv. vs. KU100
    plt.plot([2+off,3-off], [82+o,82+o], color='k')



plt.ylabel(ylabel, fontsize=ylabel_textsize)
plt.ylim([0,100.0])
plt.yticks(ticks=np.arange(0, 110, 10))
plt.tight_layout()
plt.title('Static')

plt.text(x=-0.1,y=-17,s='--Real--')
plt.text(x=1.9,y=-17,s='--Virtual--')

plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
plt.show(block=True)


ylabel = 'Vertical LCR (%)' 
savename = 'StaticVerticalLCR_DenseVsSparse.eps'
density_directions = [[7, 15, 6, 14], [1, 9, 2, 10]]
lcr_data = lcr_static_ele

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

plt.figure(figsize=(3,3))
plt.grid(axis='y')
offs = 0.25
medians = np.median(confusion_rates.T * 100.0, axis=0)

h_idx = [0,2,4,6]
plt.errorbar([1-offs, 2-offs, 3-offs, 4-offs], 
                medians[h_idx], yerr=[medians[h_idx]-median_iqr_density.low[h_idx], median_iqr_density.high[h_idx]-medians[h_idx]], 
                ls='', marker='D', markerfacecolor='white', markeredgecolor='tab:blue', capsize=3.5, color='tab:blue', label='15° spacing')
l_idx = [1,3,5,7]
plt.errorbar([1+offs, 2+offs, 3+offs, 4+offs], 
                medians[l_idx], yerr=[medians[l_idx]-median_iqr_density.low[l_idx], median_iqr_density.high[l_idx]-medians[l_idx]], 
                ls='', marker='s', markerfacecolor='white', markeredgecolor='k', capsize=3.5, color='k', label='30° spacing')

plt.xticks([1,2,3,4], xlabel)
plt.plot([1.5,1.5], [0,100], color='gray')
plt.plot([2.5,2.5], [0,100], color='gray')
plt.plot([3.5,3.5], [0,100], color='gray')
plt.plot([4.5,4.5], [0,100], color='gray')

if 0:
    # p values
    plt.plot([1-offs, 1-offs, 1+offs, 1+offs], [28,30,30,28], color='k')
    plt.text(x=1-0.075, y=30, s='*', color='k')
    plt.plot([2-offs, 2-offs, 2+offs, 2+offs], [68,70,70,68], color='k')
    plt.text(x=2-0.075*3, y=70, s='***', color='k')
    plt.plot([3-offs, 3-offs, 3+offs, 3+offs], [68,70,70,68], color='k')
    plt.text(x=3-0.075*3, y=70, s='***', color='k')
    plt.plot([4-offs, 4-offs, 4+offs, 4+offs], [78,80,80,78], color='k')
    plt.text(x=4-0.075*3, y=80, s='***', color='k')
if 1:
    # p values
    #plt.plot([1-offs, 1+offs], [30,30], color='k')
    #plt.text(x=1-0.075, y=30, s='*', color='k')
    plt.plot([2-offs, 2+offs], [70,70], color='k')
    #plt.text(x=2-0.075*3, y=70, s='***', color='k')
    plt.plot([3-offs, 3+offs], [70,70], color='k')
    #plt.text(x=3-0.075*3, y=70, s='***', color='k')
    plt.plot([4-offs, 4+offs], [80,80], color='k')
    #plt.text(x=4-0.075*3, y=80, s='***', color='k')


plt.ylabel(ylabel, fontsize=ylabel_textsize)
plt.ylim([0,100.0])
plt.yticks(ticks=np.arange(0, 110, 10))
plt.legend(framealpha=1.0)
plt.title('Static')
plt.tight_layout()

plt.text(x=-0.1 + 1, y=-17,s='--Real--')
plt.text(x=1.9 + 1, y=-17,s='--Virtual--')

plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
plt.show(block=True)


