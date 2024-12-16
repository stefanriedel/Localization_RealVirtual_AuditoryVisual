from os.path import dirname, join as pjoin
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'ErrorMetricData')
fig_dir = pjoin(root_dir, 'Figures')

# Load precomputed data
lcr_static_ele = np.load(file=pjoin(data_dir, 'LocalConfusionDataElevationStatic.npy'), allow_pickle=True).tolist()
lcr_dynamic_ele = np.load(file=pjoin(data_dir, 'LocalConfusionDataElevationDynamic.npy'), allow_pickle=True).tolist()

# Common parameters
title_textsize = 14
ylabel_textsize = 13
xlabel_textsize = 13
rot_angle = 25

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def plot_confusion_rates(ax, lcr_data, conditions, xlabel, ylabel, title):
    density_directions = [[7, 15, 6, 14], [1, 9, 2, 10]]
    confusion_rates = np.zeros((8, 16))

    condition_idx = 0
    for condition in conditions:
        for directions in density_directions:
            for subj in range(16):
                if condition.endswith('OpenEars'):
                    confusions = np.asarray(lcr_data[condition]['Confusions'])[directions, :][:, subj]
                else:
                    confusions = np.asarray(lcr_data[condition]['Confusions'])[directions, :][:, [subj, subj + 16]]
                    confusions = np.nanmean(confusions, axis=1)

                n = np.sum(~np.isnan(confusions))  # Number of local data points
                k = np.nansum(confusions)  # Number of confusion in local data points
                confusion_rates[condition_idx, subj] = k / n

            condition_idx += 1

    offs = 0.2
    medians = np.nanmedian(confusion_rates.T * 100.0, axis=0)

    quartile1, medians, quartile3 = np.nanpercentile(confusion_rates * 100.0, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(np.sort(confusion_rates * 100.0), quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    h_idx = [0, 2, 4, 6]
    x_inds = [1 - offs, 2 - offs, 3 - offs, 4 - offs]
    ax.plot(x_inds, medians[h_idx], marker='D', markerfacecolor='white', markeredgecolor='k', zorder=3, ls='')
    ax.vlines(x_inds, quartile1[h_idx], quartile3[h_idx], color='tab:blue', linestyle='-', lw=5)
    ax.vlines(x_inds, whiskersMin[h_idx], whiskersMax[h_idx], color='tab:blue', linestyle='-', lw=1)

    l_idx = [1, 3, 5, 7]
    x_inds = [1 + offs, 2 + offs, 3 + offs, 4 + offs]
    ax.plot(x_inds, medians[l_idx], marker='o', markerfacecolor='white', markeredgecolor='k', zorder=3, ls='')
    ax.vlines(x_inds, quartile1[l_idx], quartile3[l_idx], color='k', linestyle='-', lw=5)
    ax.vlines(x_inds, whiskersMin[l_idx], whiskersMax[l_idx], color='k', linestyle='-', lw=1)

    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(xlabel, fontsize=xlabel_textsize, rotation=rot_angle)
    ax.set_ylabel(ylabel, fontsize=ylabel_textsize)
    ax.set_ylim([0, 100.0])
    ax.set_yticks(ticks=np.arange(0, 110, 10))
    ax.grid(axis='y')
    ax.set_title(title, fontsize=title_textsize)

    ax.text(x=0 + 1, y=-27.5,s='--Real--', fontsize=xlabel_textsize)
    ax.text(x=2 + 1, y=-27.5,s='--Virtual--', fontsize=xlabel_textsize)

size = 1.125
# Subplot settings
fig, axs = plt.subplots(1, 2, figsize=(6*size, 3*size), sharey=False, gridspec_kw={'wspace': 0.2})

# Static plot
plot_confusion_rates(
    axs[0], lcr_static_ele,
    conditions=['StaticOpenEars', 'StaticOpenHeadphones', 'StaticIndivHRTF', 'StaticKU100HRTF'],
    xlabel=['Op.Ear ', 'Op.Hp. ', 'Indiv.', 'KU100'],
    ylabel='Local Confusion Rate (%)',
    title='(a) Vertical (Static)'
)
offs = 0.125
axs[0].plot([1-offs, 1+offs], [30,30], color='k')
axs[0].plot([2-offs, 2+offs], [70,70], color='k')
axs[0].plot([3-offs, 3+offs], [70,70], color='k')
axs[0].plot([4-offs, 4+offs], [80,80], color='k')

# Dynamic plot
plot_confusion_rates(
    axs[1], lcr_dynamic_ele,
    conditions=['DynamicOpenEars', 'DynamicOpenHeadphones', 'DynamicKEMARHRTF', 'DynamicKU100HRTF'],
    xlabel=['Op.Ear ', 'Op.Hp. ', 'KEMAR', 'KU100'],
    ylabel='',
    title='(b) Vertical (Dynamic)'
)
offs = 0.125
axs[1].plot([2-offs, 2+offs], [40,40], color='k')
axs[1].plot([3-offs, 3+offs], [70,70], color='k')
axs[1].plot([4-offs, 4+offs], [70,70], color='k')

# Legend
blue_patch = plt.Line2D([], [], color='tab:blue', marker='D', markerfacecolor='white', markeredgecolor='k', lw=3,
                        markersize=6, label='15° vertical spacing')
black_patch = plt.Line2D([], [], color='k', marker='o', markerfacecolor='white', markeredgecolor='k', lw=3,
                          markersize=6, label='30° vertical spacing')
fig.legend(handles=[blue_patch, black_patch], framealpha=1.0, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.25), fontsize=xlabel_textsize)

plt.savefig(pjoin(fig_dir, 'VerticalLCR_DenseVsSparse.eps'), bbox_inches='tight')

plt.tight_layout()
plt.show()
