import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import dirname, join as pjoin

# Set up directories (adjust paths as needed)
root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ErrorMetricData')

# Load static data
lcr_static_ele = np.load(pjoin(data_dir, 'LocalConfusionDataElevationStatic.npy'), allow_pickle=True).tolist()
lcr_static_azi = np.load(pjoin(data_dir, 'LocalConfusionDataAzimuthStatic.npy'), allow_pickle=True).tolist()

# Load dynamic data
lcr_dynamic_ele = np.load(pjoin(data_dir, 'LocalConfusionDataElevationDynamic.npy'), allow_pickle=True).tolist()
lcr_dynamic_azi = np.load(pjoin(data_dir, 'LocalConfusionDataAzimuthDynamic.npy'), allow_pickle=True).tolist()

# Conditions and labels
static_conditions = ['StaticOpenEars', 'StaticOpenHeadphones', 'StaticIndivHRTF', 'StaticKU100HRTF']
dynamic_conditions = ['DynamicOpenEars', 'DynamicOpenHeadphones', 'DynamicKEMARHRTF', 'DynamicKU100HRTF']

xlabels_static = ['Op.Ear', 'Op.Hp.', 'Indiv.', 'KU100']
xlabels_dynamic = ['Op.Ear', 'Op.Hp.', 'KEMAR', 'KU100']

# Common parameters
ylabel_textsize = 12
xlabel_textsize = 11

# Utility function to calculate confusion rates
def getConfusionRatesConstantSampleSize(lcr_data, conditions, directions):
    confusion_rates = np.zeros((len(conditions), 16))

    for condition_idx, condition in enumerate(conditions):
        for subj in range(16):
            if condition in ['StaticOpenEars', 'DynamicOpenEars']:
                confusions = np.asarray(lcr_data[condition]['Confusions'])[directions, :][:, subj]
            else:
                confusions = np.asarray(lcr_data[condition]['Confusions'])[directions, :][:, [subj, subj + 16]]
                confusions = np.nanmean(confusions, axis=1)

            n = np.sum(~np.isnan(confusions))  # Number of local data points
            k = np.nansum(confusions)  # Number of confusions in local data points
            confusion_rates[condition_idx, subj] = k / n

    return confusion_rates

# Directions for horizontal and vertical plots
horizontal_directions = list(range(20))
vertical_directions = list(range(20)) + list(range(21, 25))

# Generate confusion rates
static_horizontal_rates = getConfusionRatesConstantSampleSize(lcr_static_azi, static_conditions, horizontal_directions)
static_vertical_rates = getConfusionRatesConstantSampleSize(lcr_static_ele, static_conditions, vertical_directions)
dynamic_horizontal_rates = getConfusionRatesConstantSampleSize(lcr_dynamic_azi, dynamic_conditions, horizontal_directions)
dynamic_vertical_rates = getConfusionRatesConstantSampleSize(lcr_dynamic_ele, dynamic_conditions, vertical_directions)

# Set up the figure and subplots
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), constrained_layout=True)

# Plot Static Horizontal
sns.violinplot(data=static_horizontal_rates.T * 100, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'lightgreen', 'lightcoral'], ax=axes[0], inner_kws=dict(whis_width=1, color="black", marker=''))
axes[0].plot([0,1,2,3], np.median(static_horizontal_rates.T * 100.0, axis=0), linestyle='', marker='o', markerfacecolor='white', markeredgecolor='k')
axes[0].set_title('(a) Horizontal (Static)')
axes[0].set_xticks(range(len(xlabels_static)))
axes[0].set_xticklabels(xlabels_static, fontsize=xlabel_textsize)
axes[0].set_ylabel('Local Confusion Rate (%)', fontsize=ylabel_textsize)
axes[0].grid(axis='y')
axes[0].set_ylim(0, 100)
axes[0].set_yticks(ticks=np.arange(0, 110, 10))
# Connect all pariwise significant tests (p < 0.05) after Bonferroni correction
off = 0.075
o = -8
# OpEar vs. Rest
axes[0].plot([0+off, 1-off], [84+o,84+o], color='k')
axes[0].plot([0+off, 3-off], [86+o,86+o], color='k')
# OpHp vs. Indiv. 
axes[0].plot([1+off,2-off], [82+o,82+o], color='k')
#Report mean LCR values:
yval = 85
clr = 'dimgray'
fs=11
off = -0.1
axes[0].text(x=0 + off,y=yval+6,s=r'$\overline{\mathrm{LCR}}:$', color=clr, fontsize=fs)
axes[0].text(x=0 + off,y=yval,s='4%', color=clr, fontsize=fs)
axes[0].text(x=1 + off,y=yval,s='13%', color=clr, fontsize=fs)
axes[0].text(x=2 + off,y=yval,s='8%', color=clr, fontsize=fs)
axes[0].text(x=3 + off,y=yval,s='11%', color=clr, fontsize=fs)
axes[0].text(x=0,y=-15,s='--Real--', fontsize=xlabel_textsize)
axes[0].text(x=2,y=-15,s='--Virtual--', fontsize=xlabel_textsize)


# Plot Dynamic Horizontal
sns.violinplot(data=dynamic_horizontal_rates.T * 100, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'khaki', 'lightcoral'], ax=axes[1],inner_kws=dict(whis_width=1, color="black", marker=''))
axes[1].plot([0,1,2,3], np.median(dynamic_horizontal_rates.T * 100.0, axis=0), linestyle='', marker='o', markerfacecolor='white', markeredgecolor='k')
axes[1].set_title('(b) Horizontal (Dynamic)')
axes[1].set_xticks(range(len(xlabels_dynamic)))
axes[1].set_xticklabels(xlabels_dynamic, fontsize=xlabel_textsize)
#axes[1].set_ylabel('Local Confusion Rate (%)')
axes[1].grid(axis='y')
axes[1].set_ylim(0, 100)
axes[1].set_yticks(ticks=np.arange(0, 110, 10))

# Asterisk to indicate significant differences to static experiment
axes[1].text(x=0-0.075, y=2.5, s='*', fontsize=14)
axes[1].text(x=1-0.075, y=2.5, s='*', fontsize=14)
axes[1].text(x=3-0.075, y=2.5, s='*', fontsize=14)
#Report mean LCR values:
yval = 85
clr = 'dimgray'
fs=11
off = -0.1
axes[1].text(x=0 + off,y=yval+6,s=r'$\overline{\mathrm{LCR}}:$', color=clr, fontsize=fs)
axes[1].text(x=0 + off,y=yval,s='0%', color=clr, fontsize=fs)
axes[1].text(x=1 + off,y=yval,s='0%', color=clr, fontsize=fs)
axes[1].text(x=2 + off,y=yval,s='1%', color=clr, fontsize=fs)
axes[1].text(x=3 + off,y=yval,s='0%', color=clr, fontsize=fs)
axes[1].text(x=0,y=-15,s='--Real--', fontsize=xlabel_textsize)
axes[1].text(x=2,y=-15,s='--Virtual--', fontsize=xlabel_textsize)


# Plot Static Vertical
sns.violinplot(data=static_vertical_rates.T * 100, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'lightgreen', 'lightcoral'], ax=axes[2], inner_kws=dict(whis_width=1, color="black", marker=''))
axes[2].plot([0,1,2,3], np.median(static_vertical_rates.T * 100.0, axis=0), linestyle='', marker='o', markerfacecolor='white', markeredgecolor='k')
axes[2].set_title('(c) Vertical (Static)')
axes[2].set_xticks(range(len(xlabels_static)))
axes[2].set_xticklabels(xlabels_static, fontsize=xlabel_textsize)
#axes[2].set_ylabel('Local Confusion Rate (%)')
axes[2].grid(axis='y')
axes[2].set_ylim(0, 100)
axes[2].set_yticks(ticks=np.arange(0, 110, 10))
off = 0.075
o = -8
# OpEar vs. Rest
axes[2].plot([0+off, 1-off], [84+o,84+o], color='k')
axes[2].plot([0+off, 2-off], [86+o,86+o], color='k')
axes[2].plot([0+off, 3-off], [88+o,88+o], color='k')
# OpHp vs. KU100
axes[2].plot([1+off,3-off], [82+o,82+o], color='k')
# Indiv. vs. KU100
axes[2].plot([2+off,3-off], [80+o,80+o], color='k')
#Report mean LCR values:
yval = 85
clr = 'dimgray'
fs=11
off = -0.1
axes[2].text(x=0 + off,y=yval+6,s=r'$\overline{\mathrm{LCR}}:$', color=clr, fontsize=fs)
axes[2].text(x=0 + off,y=yval,s='12%', color=clr, fontsize=fs)
axes[2].text(x=1 + off,y=yval,s='38%', color=clr, fontsize=fs)
axes[2].text(x=2 + off,y=yval,s='35%', color=clr, fontsize=fs)
axes[2].text(x=3 + off,y=yval,s='45%', color=clr, fontsize=fs)
axes[2].text(x=0,y=-15,s='--Real--', fontsize=xlabel_textsize)
axes[2].text(x=2,y=-15,s='--Virtual--', fontsize=xlabel_textsize)

# Plot Dynamic Vertical
sns.violinplot(data=dynamic_vertical_rates.T * 100, cut=0, linewidth=1.25, palette=['skyblue', 'slateblue', 'khaki', 'lightcoral'], ax=axes[3], inner_kws=dict(whis_width=1, color="black", marker=''))
axes[3].plot([0,1,2,3], np.median(dynamic_vertical_rates.T * 100.0, axis=0), linestyle='', marker='o', markerfacecolor='white', markeredgecolor='k')
axes[3].set_title('(d) Vertical (Dynamic)')
axes[3].set_xticks(range(len(xlabels_dynamic)))
axes[3].set_xticklabels(xlabels_dynamic, fontsize=xlabel_textsize)
#axes[3].set_ylabel('Local Confusion Rate (%)')
axes[3].grid(axis='y')
axes[3].set_ylim(0, 100)
axes[3].set_yticks(ticks=np.arange(0, 110, 10))

# Connect all pariwise significant tests (p < 0.05) after Bonferroni correction
# Asterisk to indicate significant differences to static experiment
off = 0.075
o = -8
# OpEar vs. Rest
axes[3].plot([0+off, 1-off], [84+o,84+o], color='k')
axes[3].plot([0+off, 2-off], [86+o,86+o], color='k')
axes[3].plot([0+off, 3-off], [88+o,88+o], color='k')
# OpHp vs. Rest
axes[3].plot([1+off,2-off], [80+o,80+o], color='k')
axes[3].plot([1+off,3-off*2], [82+o,82+o], color='k')

axes[3].text(x=0-0.075, y=20, s='*', fontsize=14)
axes[3].text(x=1-0.075, y=35, s='*', fontsize=14)
axes[3].text(x=3-0.075, y=72, s='*', fontsize=14)
#Report mean LCR values:
yval = 85
clr = 'dimgray'
fs=11
off = -0.1
axes[3].text(x=0 + off,y=yval+6,s=r'$\overline{\mathrm{LCR}}:$', color=clr, fontsize=fs)
axes[3].text(x=0 + off,y=yval,s='4%', color=clr, fontsize=fs)
axes[3].text(x=1 + off,y=yval,s='15%', color=clr, fontsize=fs)
axes[3].text(x=2 + off,y=yval,s='35%', color=clr, fontsize=fs)
axes[3].text(x=3 + off,y=yval,s='38%', color=clr, fontsize=fs)
axes[3].text(x=0,y=-15,s='--Real--', fontsize=xlabel_textsize)
axes[3].text(x=2,y=-15,s='--Virtual--', fontsize=xlabel_textsize)

# Save and show the figure
plt.savefig(pjoin('Figures', 'StaticDynamicLCR.eps'), bbox_inches='tight')
plt.show()
