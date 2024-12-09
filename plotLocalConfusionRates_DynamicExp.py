from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ErrorMetricData')
fig_dir = pjoin(root_dir, 'Figures')

# LOAD ALL PRECOMPUTED DATA
lcr_dynamic_ele = np.load(file=pjoin(data_dir, 'LocalConfusionDataElevationDynamic.npy'), allow_pickle=True).tolist()
lcr_dynamic_azi = np.load(file=pjoin(data_dir, 'LocalConfusionDataAzimuthDynamic.npy'), allow_pickle=True).tolist()
slope_dynamic_ele = np.load(file=pjoin(data_dir, 'SlopeDataElevationDynamic.npy'), allow_pickle=True).tolist()
slope_dynamic_azi = np.load(file=pjoin(data_dir, 'SlopeDataAzimuthDynamic.npy'), allow_pickle=True).tolist()

ylabel_textsize = 10

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



conditions = ['DynamicOpenEars', 'DynamicOpenHeadphones', 'DynamicKEMARHRTF', 'DynamicKU100HRTF']
xlabel = ['Op.Ear ', ' Op.Hp. ', ' KEMAR' , ' KU100']


# DYNAMIC HORIZONTAL: GLOBAL
directions = [*range(20)] 
#ylabel = 'Horizontal LCR (%)'
ylabel = 'Local Confusion Rate (%)' 
savename = 'DynamicHorizontalLCR.eps'

confusion_rates = getConfusionRatesConstantSampleSize(lcr_dynamic_azi, conditions, directions)

plt.figure(figsize=(3,3))
plt.grid(axis='y')
ax = sns.violinplot(confusion_rates.T * 100.0, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'khaki', 'lightcoral'], inner_kws=dict(whis_width=1, color="black", marker=''))
plt.plot([0,1,2,3], np.median(confusion_rates.T * 100.0, axis=0), linestyle='', marker='o', markerfacecolor='white', markeredgecolor='k')

#plt.setp(ax.collections, alpha=.1)
plt.xticks([0,1,2,3], xlabel)

off = 0.075
# OpEar vs. OpHp.
#plt.plot([0+off,0+off,1-off,1-off], [8,10,10,8], color='k')
#plt.text(x=0.5-0.075*2, y=11, s='ns')

# plt.plot([1+off,1+off,2-off,2-off], [48,50,50,48], color='k')
# plt.text(x=1.5-0.075*2, y=51, s='**')

# plt.plot([2+off,2+off,3-off,3-off], [48,50,50,48], color='k')
# plt.text(x=2.5-0.075*2, y=51, s='ns')

# plt.plot([1+off*2,1+off*2,3-off*2,3-off*2], [38,40,40,38], color='k')
# plt.text(x=2-0.075*2, y=41, s='ns')

# Alternative: Connect all pariwise significant tests (p < 0.05) after Bonferroni correction
# Asterisk to indicate significant differences to static experiment
if 1:
    plt.text(x=0-0.1, y=2.5, s='*', fontsize=14)
    plt.text(x=1-0.1, y=2.5, s='*', fontsize=14)
    plt.text(x=3-0.1, y=2.5, s='*', fontsize=14)



plt.ylabel(ylabel, fontsize=ylabel_textsize)
plt.ylim([0,100.0])
#plt.ylim([-15,100.0])
plt.yticks(ticks=np.arange(0, 110, 10))
plt.tight_layout()
#plt.title('Dynamic')
plt.title('(b) Horizontal (Dynamic)')

#Report mean LCR values:
yval = 85
clr = 'dimgray'
fs=10
off = -0.1
plt.text(x=0 + off,y=yval+6,s=r'$\overline{\mathrm{LCR}}:$', color=clr, fontsize=fs)
plt.text(x=0 + off,y=yval,s='0%', color=clr, fontsize=fs)
plt.text(x=1 + off,y=yval,s='0%', color=clr, fontsize=fs)
plt.text(x=2 + off,y=yval,s='1%', color=clr, fontsize=fs)
plt.text(x=3 + off,y=yval,s='0%', color=clr, fontsize=fs)

plt.text(x=-0.1,y=-17,s='--Real--')
plt.text(x=1.9,y=-17,s='--Virtual--')

plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
plt.show(block=False)


# DYNAMIC VERTICAL: GLOBAL
directions = [*range(20)] + [*range(21, 25)]
#ylabel = 'Vertical LCR (%)' 
ylabel = 'Local Confusion Rate (%)' 
savename = 'DynamicVerticalLCR.eps'

confusion_rates = getConfusionRatesConstantSampleSize(lcr_dynamic_ele, conditions, directions)

plt.figure(figsize=(3,3))
plt.grid(axis='y')
sns.violinplot(confusion_rates.T * 100.0, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'khaki', 'lightcoral'], inner_kws=dict(whis_width=1, color="black", marker=''))
plt.plot([0,1,2,3], np.median(confusion_rates.T * 100.0, axis=0), linestyle='', marker='o', markerfacecolor='white', markeredgecolor='k')
plt.xticks([0,1,2,3], xlabel)

off = 0.075
# OpEars vs. OpHp
#plt.plot([0+off,0+off,1-off,1-off], [38,40,40,38], color='k')
#plt.text(x=0.5-0.075*2, y=40, s='**')

# plt.plot([1+off,1+off,2-off,2-off], [88,90,90,88], color='k')
# plt.text(x=1.5-0.075*2, y=91, s='ns')

# plt.plot([2+off,2+off,3-off,3-off], [88,90,90,88], color='k')
# plt.text(x=2.5-0.075*2, y=90, s='**')

# plt.plot([1+off*2,1+off*2,3-off*2,3-off*2], [78,80,80,78], color='k')
# plt.text(x=2-0.075, y=80, s='*')

# Alternative: Connect all pariwise significant tests (p < 0.05) after Bonferroni correction
# Asterisk to indicate significant differences to static experiment
if 1:
    o = -8
    # OpEar vs. Rest
    plt.plot([0+off, 1-off], [84+o,84+o], color='k')
    plt.plot([0+off, 2-off], [86+o,86+o], color='k')
    plt.plot([0+off, 3-off], [88+o,88+o], color='k')
    # OpHp vs. Rest
    plt.plot([1+off,2-off], [80+o,80+o], color='k')
    plt.plot([1+off,3-off*2], [82+o,82+o], color='k')

    
    plt.text(x=0-0.1, y=20, s='*', fontsize=14)
    plt.text(x=1-0.1, y=35, s='*', fontsize=14)
    plt.text(x=3-0.1, y=72, s='*', fontsize=14)

    #plt.text(x=3-0.075, y=72.5, s='*', fontsize=12)



plt.ylabel(ylabel, fontsize=ylabel_textsize)
#plt.ylim([0,100.0])
plt.ylim([0,100.0])
plt.yticks(ticks=np.arange(0, 110, 10))
plt.tight_layout()
#plt.title('Dynamic')
plt.title('(d) Vertical (Dynamic)')

#Report mean LCR values:
yval = 85
clr = 'dimgray'
fs=10
off = -0.1
plt.text(x=0 + off,y=yval+6,s=r'$\overline{\mathrm{LCR}}:$', color=clr, fontsize=fs)
plt.text(x=0 + off,y=yval,s='4%', color=clr, fontsize=fs)
plt.text(x=1 + off,y=yval,s='15%', color=clr, fontsize=fs)
plt.text(x=2 + off,y=yval,s='35%', color=clr, fontsize=fs)
plt.text(x=3 + off,y=yval,s='38%', color=clr, fontsize=fs)

plt.text(x=-0.1,y=-17,s='--Real--')
plt.text(x=1.9,y=-17,s='--Virtual--')

plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
plt.show(block=False)


#ylabel = 'Vertical LCR (%)' 
ylabel = 'Local Confusion Rate (%)' 
savename = 'DynamicVerticalLCR_DenseVsSparse.eps'
density_directions = [[7, 15, 6, 14], [1, 9, 2, 10]]
lcr_data = lcr_dynamic_ele

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

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

plt.figure(figsize=(3,3))
plt.grid(axis='y')
offs = 0.25
medians = np.nanmedian(confusion_rates.T * 100.0, axis=0)


quartile1, medians, quartile3 = np.nanpercentile(confusion_rates * 100.0, [25, 50, 75], axis=1)
whiskers = np.array([
    adjacent_values(sorted_array, q1, q3)
    for sorted_array, q1, q3 in zip(np.sort(confusion_rates * 100.0), quartile1, quartile3)])
whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

h_idx = [0,2,4,6]
# plt.errorbar([1-offs, 2-offs, 3-offs, 4-offs], 
#                 medians[h_idx], yerr=[medians[h_idx]-median_iqr_density.low[h_idx], median_iqr_density.high[h_idx]-medians[h_idx]], 
#                 ls='', marker='D', markerfacecolor='white', markeredgecolor='tab:blue', capsize=3.5, color='tab:blue', label='15° spacing')
x_inds = [1-offs, 2-offs, 3-offs, 4-offs]
# violins = plt.violinplot(
#         confusion_rates[h_idx,:].T * 100.0, positions=x_inds, showmeans=False, showmedians=False,
#         showextrema=False)
# for pc in violins['bodies']:
#     pc.set_facecolor('tab:blue')
#     pc.set_edgecolor('black')
#     pc.set_alpha(0.1)
plt.plot(x_inds, medians[h_idx], marker='D', markerfacecolor='white', markeredgecolor='k', zorder=3, ls='')
plt.vlines(x_inds, quartile1[h_idx], quartile3[h_idx], color='tab:blue', linestyle='-', lw=5)
plt.vlines(x_inds, whiskersMin[h_idx], whiskersMax[h_idx], color='tab:blue', linestyle='-', lw=1)


l_idx = [1,3,5,7]
# plt.errorbar([1+offs, 2+offs, 3+offs, 4+offs], 
#                 medians[l_idx], yerr=[medians[l_idx]-median_iqr_density.low[l_idx], median_iqr_density.high[l_idx]-medians[l_idx]], 
#                 ls='', marker='s', markerfacecolor='white', markeredgecolor='k', capsize=3.5, color='k', label='30° spacing')
x_inds = [1+offs, 2+offs, 3+offs, 4+offs]
# violins = plt.violinplot(
#         confusion_rates[l_idx,:].T * 100.0, positions=x_inds, showmeans=False, showmedians=False,
#         showextrema=False)
# for pc in violins['bodies']:
#     pc.set_facecolor('k')
#     pc.set_edgecolor('black')
#     pc.set_alpha(0.1)
plt.plot(x_inds, medians[l_idx], marker='o', markerfacecolor='white', markeredgecolor='k', zorder=3, ls='')
plt.vlines(x_inds, quartile1[l_idx], quartile3[l_idx], color='k', linestyle='-', lw=5)
plt.vlines(x_inds, whiskersMin[l_idx], whiskersMax[l_idx], color='k', linestyle='-', lw=1)

plt.xticks([1,2,3,4], xlabel)
plt.plot([1.5,1.5], [0,100], color='gray')
plt.plot([2.5,2.5], [0,100], color='gray')
plt.plot([3.5,3.5], [0,100], color='gray')
plt.plot([4.5,4.5], [0,100], color='gray')


if 0:
    # p values
    plt.plot([1-offs, 1-offs, 1+offs, 1+offs], [28,30,30,28], color='k')
    plt.text(x=1-0.075*2, y=31, s='ns', color='k')
    plt.plot([2-offs, 2-offs, 2+offs, 2+offs], [68,70,70,68], color='k')
    plt.text(x=2-0.075*3, y=70, s='***', color='k')
    plt.plot([3-offs, 3-offs, 3+offs, 3+offs], [68,70,70,68], color='k')
    plt.text(x=3-0.075*3, y=70, s='***', color='k')
    plt.plot([4-offs, 4-offs, 4+offs, 4+offs], [78,80,80,78], color='k')
    plt.text(x=4-0.075*2, y=80, s='**', color='k')
if 1:
    offs = 0.15
    # p values
    #plt.plot([1-offs, 1+offs], [30,30], color='k')
    #plt.text(x=1-0.075*2, y=31, s='ns', color='k')
    plt.plot([2-offs, 2+offs], [70,70], color='k')
    #plt.text(x=2-0.075*3, y=70, s='***', color='k')
    plt.plot([3-offs, 3+offs], [70,70], color='k')
    #plt.text(x=3-0.075*3, y=70, s='***', color='k')
    plt.plot([4-offs, 4+offs], [80,80], color='k')
    #plt.text(x=4-0.075*2, y=80, s='**', color='k')

import matplotlib.patches as mpatches
import matplotlib.lines as mlines

blue_patch = mlines.Line2D([], [], color='tab:blue', marker='D', markerfacecolor='white', markeredgecolor='k', lw=3,
                          markersize=5, label='15° spacing')
black_patch = mlines.Line2D([], [], color='k', marker='o', markerfacecolor='white', markeredgecolor='k', lw=3,
                          markersize=5, label='30° spacing')

plt.ylabel(ylabel, fontsize=ylabel_textsize)
plt.ylim([0,100.0])
plt.yticks(ticks=np.arange(0, 110, 10))
plt.legend(handles=[blue_patch, black_patch],framealpha=1.0)
plt.title('(b) Vertical (Dynamic)')
plt.tight_layout()

plt.text(x=-0.1 + 1, y=-17,s='--Real--')
plt.text(x=1.9 + 1, y=-17,s='--Virtual--')

plt.savefig(pjoin(fig_dir, savename), bbox_inches='tight')
plt.show(block=True)


