import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin
from matplotlib import cm
import matplotlib.colors as colors
from Utility.pairwiseTests import posthoc_wilcoxon
from scipy.spatial.distance import cdist, euclidean
import warnings
import scipy.stats as stats

# Render vertical plane plots
name_list = [
    '0DEG', '-30DEG', '30DEG', '-90DEG', '90DEG', '180DEG', '-150DEG', '150DEG'
]
# deg_list = ['0 deg.', '-30 deg.', '30 deg.', '-90 deg.',
#            '90 deg.', '180 deg.', '-150 deg.', '150 deg.']
deg_list = [
    r'$0$' + '°', r'$-30$' + '°', r'$30$' + '°', r'$-90$' + '°', r'$90$' + '°',
    r'$180$' + '°', r'$-150$' + '°', r'$150$' + '°'
]
idcs_list = [
    np.array([25, 1, 24, 9, 17]) - 1,  # 0 DEG
    np.array([8, 23, 16]) - 1,  # -30 DEG
    np.array([2, 10]) - 1,  # 30 DEG
    np.array([7, 22, 15, 20]) - 1,  # -90 DEG
    np.array([3, 11, 18]) - 1,  # 90 DEG
    np.array([5, 13, 19]) - 1,  # 180 DEG
    np.array([6, 14]) - 1,  # -150 DEG
    np.array([4, 12]) - 1  # 150 DEG
]
pairtest_list = [
    [[0, 1], [1, 2], [2, 3], [3, 4]],
    [[0, 1], [1, 2]],
    [[0, 1]],
    [[0, 1], [1, 2], [2, 3]],
    [[0, 1], [1, 2]],
    [[0, 1], [1, 2]],
    [[0, 1]],
    [[0, 1]],
]

target_ele_list = [
    np.array([-15, 0, 15, 30, 60]),
    np.array([0, 15, 30]),
    np.array([0, 30]),
    np.array([0, 15, 30, 60]),
    np.array([0, 30, 60]),
    np.array([0, 30, 60]),
    np.array([0, 30]),
    np.array([0, 30])
]

# For error metric table
NUM_CHANNELS = 25
L1_idcs = np.arange(8)
L2_idcs = np.arange(8, 16)
L3_idcs = np.arange(16, 20)
dir_sets = [
    L1_idcs,
    np.array([0, 24, 23]),
    np.concatenate((L2_idcs, L3_idcs)),
]
dirset_names = [
    'Horizontal', 'Frontal', 'Elevated'
]

title_bool_list = [True, False, True, False, True, False, False, True]

all_colors = ['tab:orange'] * 8 + ['tab:red'] * 8 + ['tab:purple'] * 4 + [
    'tab:brown'
] + ['tab:green'] * 3 + ['tab:blue']


def MAD(data):
    return np.nanmedian(np.abs(data - np.nanmedian(data)))


def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1


def convertAziEleToPlane(azi, ele):
    """ Convert azi and ele values to XY-values in 2D map visualization. """
    coord = np.array(
        [np.asarray(azi, dtype=float),
         np.asarray(ele, dtype=float)]).transpose()
    coord[:, 1] = (90.0 - coord[:, 1]) / 90.0  # convert to r
    coord[:, 0] = (coord[:, 0] + 90.0) / 180.0 * np.pi  # convert to radians
    coord_x = np.cos(coord[:, 0]) * coord[:, 1]
    coord_y = np.sin(coord[:, 0]) * coord[:, 1]
    return coord_x, coord_y


def computeAndSaveErrorMetrics(save_dir, EXP, dir_sets, dirset_names,
                               final_dict_names, local_azi_ele_data,
                               targets_azi_ele):
    NUM_CHANNELS = 25

    # dir_sets = [L1_idcs, np.array([0, 24, 23]), np.concatenate((L2_idcs, L3_idcs))]
    for dirs, dirset_name in zip(dir_sets, dirset_names):
        local_azi_error = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }
        local_ele_error = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }
        local_azi_error_rms = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }
        local_ele_error_rms = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }
        local_azi_bias = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }
        local_ele_bias = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }

        for dict_name in final_dict_names:
            num_trials = len(local_azi_ele_data[dict_name])
            azi_errors = []
            ele_errors = []

            azi_bias = []
            ele_bias = []

            for tr in range(NUM_CHANNELS):
                if num_trials == NUM_CHANNELS:
                    responses_azi_ele = local_azi_ele_data[dict_name][tr]
                else:
                    first_resp = local_azi_ele_data[dict_name][tr]
                    doubled_resp = local_azi_ele_data[dict_name][tr +
                                                                 NUM_CHANNELS]
                    responses_azi_ele = np.concatenate(
                        (first_resp, doubled_resp))

                nan_angles = np.array([np.nan, np.nan])
                if np.abs(targets_azi_ele[tr % NUM_CHANNELS, 0]) < 89.0:
                    fb_confusion_idcs = np.where(
                        np.abs(responses_azi_ele[:, 0]) > 90.0)[0]
                    responses_azi_ele[fb_confusion_idcs, :] = nan_angles[
                        None, :]
                if np.abs(targets_azi_ele[tr % NUM_CHANNELS, 0]) > 91.0:
                    fb_confusion_idcs = np.where(
                        np.abs(responses_azi_ele[:, 0]) < 90.0)[0]
                    responses_azi_ele[fb_confusion_idcs, :] = nan_angles[
                        None, :]

                azi_error_cand = targets_azi_ele[tr % NUM_CHANNELS,
                                                 0] - responses_azi_ele[:, 0]
                azi_error_cand[azi_error_cand > 180.0] = 360.0 - \
                    azi_error_cand[azi_error_cand > 180.0]
                azi_error_cand[azi_error_cand < -180.0] = 360.0 + \
                    azi_error_cand[azi_error_cand < -180.0]

                ele_error_cand = targets_azi_ele[tr % NUM_CHANNELS,
                                                 1] - responses_azi_ele[:, 1]

                azi_errors.append(azi_error_cand)
                ele_errors.append(ele_error_cand)

            azi_errors = np.asarray(azi_errors)
            ele_errors = np.asarray(ele_errors)
            local_azi_error[dict_name] = azi_errors
            local_ele_error[dict_name] = ele_errors

            dirs = np.asarray(dirs)
            dirs = dirs[dirs != 20]  # Remove zenith direction
            azi_errors = azi_errors[dirs, :]
            ele_errors = ele_errors[dirs, :]

            # MEAN_BASED_METRICS = True
            # if MEAN_BASED_METRICS:  # RMS over directions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if num_trials == NUM_CHANNELS:
                    azi_errors_squared = azi_errors**2
                    ele_errors_squared = ele_errors**2
                else:
                    first_azi_error = azi_errors[:, :16]
                    doubled_azi_error = azi_errors[:, 16:]
                    stacked_azi_errors = np.array(
                        [first_azi_error, doubled_azi_error])
                    azi_errors = np.nanmean(stacked_azi_errors, axis=0)
                    azi_errors_squared = np.nanmean(stacked_azi_errors**2,
                                                    axis=0)

                    first_ele_error = ele_errors[:, :16]
                    doubled_ele_error = ele_errors[:, 16:]
                    stacked_ele_errors = np.array(
                        [first_ele_error, doubled_ele_error])
                    ele_errors = np.nanmean(stacked_ele_errors, axis=0)
                    ele_errors_squared = np.nanmean(stacked_ele_errors**2,
                                                    axis=0)

                azi_error_rms = np.sqrt(np.nanmean(azi_errors_squared, axis=0))
                ele_error_rms = np.sqrt(np.nanmean(ele_errors_squared, axis=0))
                azi_bias = np.abs(np.nanmean(azi_errors, axis=0))
                ele_bias = np.abs(np.nanmean(ele_errors, axis=0))
            local_azi_error_rms[dict_name] = azi_error_rms
            local_ele_error_rms[dict_name] = ele_error_rms
            local_azi_bias[dict_name] = azi_bias
            local_ele_bias[dict_name] = ele_bias

        # Save to disk with dirset name and dictionary name
        metric_data_list = [
            local_azi_error_rms, local_azi_bias, local_ele_error_rms,
            local_ele_bias
        ]
        metric_name_list = [
            'AzimuthError', 'AzimuthBias', 'ElevationError', 'ElevationBias'
        ]
        for metric_data, metric_name in zip(metric_data_list,
                                            metric_name_list):
            np.save(file=pjoin(
                save_dir,
                EXP + '_' + dirset_name + '_' + metric_name + '.npy'),
                arr=metric_data,
                allow_pickle=True)
    return


def loadAndPrintErrorMetric(load_dir, dirset_names):
    EXPS = ['Static', 'Dynamic']
    EXP = 'Static'
    metric_name_list = [
        'AzimuthError', 'AzimuthBias', 'ElevationError', 'ElevationBias'
    ]

    print_name_list = [
        'Azimuth Error', 'Azimuth Bias', 'Elevation Error', 'Elevation Bias'
    ]

    for dirset_name in dirset_names:
        for metric_name, print_name in zip(metric_name_list, print_name_list):
            for EXP in EXPS:
                metric_data_all = np.load(file=pjoin(
                    load_dir,
                    EXP + '_' + dirset_name + '_' + metric_name + '.npy'),
                    allow_pickle=True)
                metric_data_all = metric_data_all.tolist()

                if EXP == 'Static':
                    dict_names = [
                        'StaticOpenEars', 'StaticOpenHeadphones',
                        'StaticIndivHRTF', 'StaticKU100HRTF'
                    ]
                if EXP == 'Dynamic':
                    dict_names = [
                        'DynamicOpenEars', 'DynamicOpenHeadphones',
                        'DynamicKEMARHRTF', 'DynamicKU100HRTF'
                    ]

                for dict_name in dict_names:
                    ref_name = EXP + 'OpenEars'
                    metric_data = metric_data_all[dict_name]
                    metric_data_ref = metric_data_all[ref_name]

                    significant = False
                    if dict_name != ref_name:
                        data = np.array([metric_data, metric_data_ref])
                        pairs = [[0, 1]]
                        pval, effect_size = posthoc_wilcoxon(
                            data,
                            pairs,
                            alternative_h='two-sided',
                            p_adjust=None)
                        significant = pval < 0.05
                        strong_effect = np.abs(effect_size) > 0.75

                    if significant and ~strong_effect and dict_name != ref_name:
                        med_mad_string = '\\cellcolor{lightgray!50} ' "%.1f" % np.nanmedian(
                            metric_data) + ' $\pm$ ' + "%.1f" % MAD(
                                metric_data) + ' \\,'
                    elif significant and strong_effect and dict_name != ref_name:
                        med_mad_string = '\\cellcolor{lightgray!50} ' + '\\textbf{' + "%.1f" % np.nanmedian(
                            metric_data) + ' $\pm$ ' + "%.1f" % MAD(
                                metric_data) + '}' + ' \\,'
                    else:
                        med_mad_string = "%.1f" % np.nanmedian(
                            metric_data) + ' $\pm$ ' + "%.1f" % MAD(
                                metric_data)

                    if dict_name == ref_name and EXP == 'Static':
                        print(print_name + ' ($^\circ$)' + ' & ' +
                              med_mad_string,
                              end="")
                    elif dict_name == 'StaticKU100HRTF':
                        print(' & ' + med_mad_string + ' & ')
                    elif dict_name == 'DynamicKU100HRTF':
                        print(' & ' + med_mad_string + ' \\\\')
                    else:
                        print(' & ' + med_mad_string, end="")
        print('\n\n')


def computeMetrics(response_elevations,
                   target_elevations,
                   median_statistic=True):
    N = response_elevations.shape[1]
    if not median_statistic:
        response_elevations = response_elevations.flatten()
        target_elevations = np.tile(target_elevations,
                                    reps=(N, 1)).flatten('F')
        non_nan_idcs = ~np.isnan(response_elevations)

        res = stats.linregress(target_elevations[non_nan_idcs],
                               response_elevations[non_nan_idcs])
        slope = res.slope
        intercept = res.intercept

        sigma = np.sqrt(
            np.mean((target_elevations[non_nan_idcs] -
                     response_elevations[non_nan_idcs])**2))

        bias = np.mean((response_elevations[non_nan_idcs] -
                        target_elevations[non_nan_idcs]))
    else:
        slope = np.zeros(N)
        slope[:] = np.nan
        intercept = np.zeros(N)
        intercept[:] = np.nan
        bias = np.zeros(N)
        bias[:] = np.nan
        sigma = np.zeros(N)
        sigma[:] = np.nan
        for n in range(N):
            resp_elevations = response_elevations[:, n]
            non_nan_idcs = ~np.isnan(resp_elevations)

            if not target_elevations[non_nan_idcs].size >= 1:
                continue

            res = stats.linregress(target_elevations[non_nan_idcs],
                                   resp_elevations[non_nan_idcs])
            slope[n] = res.slope
            intercept[n] = res.intercept

            sigma[n] = np.sqrt(
                np.mean((target_elevations[non_nan_idcs] -
                         resp_elevations[non_nan_idcs])**2))

            bias[n] = np.mean((resp_elevations[non_nan_idcs] -
                               target_elevations[non_nan_idcs]))
        slope = np.nanmedian(slope)
        intercept = np.nanmedian(intercept)
        bias = np.nanmedian(bias)
        sigma = np.nanmedian(sigma)
    return slope, intercept, sigma, bias


def renderInsetAxis(ax, active_idcs, coord_x, coord_y, pos_size_list, mkr_size=15):
    # inset axes for pictogram of source directions
    ring_color = 'grey'
    figsize = 3
    offs = 0.04
    r = 0.8
    # x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original axis
    # xlim=(x1, x2), ylim=(y1, y2),
    axins = ax.inset_axes(
        # [0.58, 0.01, 0.4, 0.4],
        # [0.02, 0.585, 0.4, 0.4],
        pos_size_list,
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
        xticks=[],
        yticks=[],
        xticklabels=[],
        yticklabels=[])

    # L1 ring
    phi = np.linspace(0, 2 * np.pi, 100)
    axins.plot(r * np.cos(phi),
               r * np.sin(phi),
               alpha=0.3,
               color=ring_color,
               zorder=5)
    # L2 ring
    axins.plot(r * np.cos(phi) * 2 / 3,
               r * np.sin(phi) * 2 / 3,
               alpha=0.3,
               color=ring_color,
               zorder=5)
    # L3 ring
    axins.plot(r * np.cos(phi) * 1 / 3,
               r * np.sin(phi) * 1 / 3,
               alpha=0.3,
               color=ring_color,
               zorder=5)
    # Loudspeaker coordinates
    plot_colors = [all_colors[i] for i in active_idcs]
    markeredge_colors = plot_colors
    axins.scatter(r * coord_x[active_idcs],
                  r * coord_y[active_idcs],
                  facecolors=markeredge_colors,
                  edgecolors='k',
                  s=mkr_size,
                  alpha=1,
                  zorder=6)
    axins.set_aspect('equal')


def plotVerticalPlanes(idcs_list, pairtest_list, target_ele_list, name_list,
                       deg_list, title_bool_list, titles, final_dict_names,
                       local_azi_ele_data, coord_x, coord_y, all_colors, EXP,
                       root_dir, plot_avg_ele):
    for mvp_idcs, pairs_to_be_tested, target_elevations, name, deg, title_bool in zip(
            idcs_list, pairtest_list, target_ele_list, name_list, deg_list,
            title_bool_list):
        num_rows = 1
        num_cols = 4
        figsize = 3
        fig, axs = plt.subplots(nrows=num_rows,
                                ncols=num_cols,
                                sharey=True,
                                figsize=(4 * figsize, 1 * figsize),
                                gridspec_kw={
                                    'hspace': 0.1,
                                    'wspace': 0.05
                                })

        if EXP == 'Static':
            conditions = final_dict_names[1:]
        if EXP == 'Dynamic':
            conditions = [
                final_dict_names[0], final_dict_names[1], final_dict_names[3],
                final_dict_names[2]
            ]

        ext = []
        for col, condition in zip(range(num_cols), conditions):
            median_ratings = np.zeros(target_elevations.size)
            elevation_ratings = []

            # Significance tests
            conditions_ele = np.asarray(
                local_azi_ele_data[condition])[mvp_idcs, :, 1]
            if col >= 1:
                cond_ele_first = np.asarray(
                    local_azi_ele_data[condition])[mvp_idcs, :, 1]
                cond_ele_second = np.asarray(
                    local_azi_ele_data[condition])[mvp_idcs + 25, :, 1]
                stacked = np.array([cond_ele_first, cond_ele_second])
                conditions_ele = np.nanmean(stacked, axis=0)

            slope, intercept, sigma, bias = computeMetrics(conditions_ele,
                                                           target_elevations,
                                                           median_statistic=True)
            axs[col].text(x=55,
                          y=-6,
                          s=r'$g = $' + str(round(slope, 2)),
                          fontsize=10)
            axs[col].text(x=55,
                          y=-14.5,
                          s=r'$\sigma = $' + str(round(sigma, 1)) + '°',
                          fontsize=10)
            x = np.linspace(-30, 90, 100)
            axs[col].plot(x, slope*x + intercept,
                          zorder=0, color='gray', ls=':')
            # axs[col].text(x=55,
            #              y=-19.5,
            #              s=r'$b = $' + str(round(bias, 1)) + '°')

            data = conditions_ele
            pvals, effect_sizes = posthoc_wilcoxon(data,
                                                   pairs_to_be_tested,
                                                   alternative_h='two-sided',
                                                   p_adjust=None)
            significant = pvals < 0.05

            if 0:
                print(condition + ' pvals: ' + str(pvals))
                print(condition + ' effect: ' + str(effect_sizes))
                print('')

            for i in range(mvp_idcs.size):
                elevation_ratings = local_azi_ele_data[condition][
                    mvp_idcs[i]][:, 1]
                if col >= 1:
                    elevation_ratings = np.concatenate(
                        (elevation_ratings,
                         local_azi_ele_data[condition][mvp_idcs[i] + 25][:,
                                                                         1]))
                if plot_avg_ele:  # PLOT MEAN DATA OF DOUBLED RESPONSES
                    elevation_ratings = conditions_ele[i, :]
                median_ratings[i] = np.nanmedian(elevation_ratings)
                axs[col].scatter(
                    target_elevations[i] +
                    (np.random.rand(elevation_ratings.size) - 0.5) * 7.5,
                    elevation_ratings,
                    alpha=0.7,
                    edgecolors='k',
                    s=15,
                    zorder=2,
                    c=all_colors[mvp_idcs[i]])
            for i in range(len(pairs_to_be_tested)):
                if True:  # significant[i] == True:
                    if pvals[i] >= 0.05:
                        pval_str = 'ns'
                        offs = -2
                        voff = 2
                    if pvals[i] < 0.05:
                        pval_str = '*'
                        offs = 0
                        voff = 0
                    if pvals[i] < 0.01:
                        pval_str = '**'
                        offs = -1.75
                        voff = 0
                    if pvals[i] < 0.001:
                        pval_str = '***'
                        offs = -2.75
                        voff = 0
                    brace_start = target_elevations[pairs_to_be_tested[i][0]]
                    brace_end = target_elevations[pairs_to_be_tested[i][1]]
                    if pairs_to_be_tested[i][1] - pairs_to_be_tested[i][0] > 1:
                        yoff = 10
                    else:
                        yoff = 0

                    axs[col].text((brace_start + brace_end) / 2 - 2 + offs,
                                  -31.5 + voff + yoff,
                                  s=pval_str,
                                  fontsize=9)

                    if np.abs(effect_sizes[i]) >= 0.75:
                        axs[col].text((brace_start + brace_end) / 2 - 2,
                                      -22.5 + yoff,
                                      s=r'$\Delta$',
                                      fontsize=8)
                    # elif np.abs(effect_sizes[i]) >= 0.5:
                    #     axs[col].text((brace_start + brace_end) / 2 - 2,
                    #                   -22.5 + yoff,
                    #                   s=r'$\Delta$',
                    #                   fontsize=8)
                    axs[col].plot([brace_start + 2.5, brace_end - 2.5],
                                  [-25 + yoff, -25 + yoff],
                                  color='k')
                    axs[col].plot([brace_start + 2.5, brace_start + 2.5],
                                  [-25 + yoff, -22.5 + yoff],
                                  color='k')
                    axs[col].plot([brace_end - 2.5, brace_end - 2.5],
                                  [-25 + yoff, -22.5 + yoff],
                                  color='k')

            axs[col].plot([-30, 90], [-30, 90], color='grey', zorder=1)
            axs[col].scatter(target_elevations,
                             median_ratings,
                             marker='D',
                             edgecolors='k',
                             linewidth=1,
                             facecolors='white',
                             zorder=3,
                             s=30)
            axs[col].set_xlim(-30, 90)
            axs[col].set_ylim(-30, 90)
            axs[col].set_xlabel('Elevation Target (deg.)')
            if col == 0:
                axs[col].set_ylabel('Elevation Response (deg.)')
                # axs[col].text(22.5, 80, s='AZI = ' + deg, fontsize=11)

                # inset axes for pictogram of source directions
                ring_color = 'grey'
                figsize = 3
                offs = 0.04
                r = 0.8
                # x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9  # subregion of the original axis
                # xlim=(x1, x2), ylim=(y1, y2),
                axins = axs[col].inset_axes(
                    # [0.58, 0.01, 0.4, 0.4],
                    [0.02, 0.585, 0.4, 0.4],
                    xlim=(-1.05, 1.05),
                    ylim=(-1.05, 1.05),
                    xticks=[],
                    yticks=[],
                    xticklabels=[],
                    yticklabels=[])

                # L1 ring
                phi = np.linspace(0, 2 * np.pi, 100)
                axins.plot(r * np.cos(phi),
                           r * np.sin(phi),
                           alpha=0.3,
                           color=ring_color,
                           zorder=5)
                # L2 ring
                axins.plot(r * np.cos(phi) * 2 / 3,
                           r * np.sin(phi) * 2 / 3,
                           alpha=0.3,
                           color=ring_color,
                           zorder=5)
                # L3 ring
                axins.plot(r * np.cos(phi) * 1 / 3,
                           r * np.sin(phi) * 1 / 3,
                           alpha=0.3,
                           color=ring_color,
                           zorder=5)
                # Loudspeaker coordinates
                plot_colors = [all_colors[i] for i in mvp_idcs]
                markeredge_colors = plot_colors
                axins.scatter(r * coord_x[mvp_idcs],
                              r * coord_y[mvp_idcs],
                              facecolors=markeredge_colors,
                              edgecolors='k',
                              s=20,
                              alpha=1,
                              zorder=6)
                axins.set_aspect('equal')
            axs[col].set_xticks([-15, 0, 15, 30, 60])
            axs[col].set_yticks([-15, 0, 15, 30, 60])
            if title_bool:
                axs[col].set_title(titles[col])
            axs[col].grid(True)
            axs[col].set_aspect('equal')
            ext.append([
                axs[col].get_window_extent().x0,
                axs[col].get_window_extent().width
            ])

        # from the axes bounding boxes calculate the optimal position of the column spanning title
        inv = fig.transFigure.inverted()
        width_left = ext[0][0] + (ext[1][0] + ext[1][1] - ext[0][0]) / 2.
        left_center = inv.transform((width_left, 1))
        width_right = ext[2][0] + (ext[3][0] + ext[3][1] - ext[2][0]) / 2.
        right_center = inv.transform((width_right, 1))

        if title_bool:
            plt.figtext(left_center[0],
                        0.98,
                        "---- Real (" + EXP + ") ----",
                        va="center",
                        ha="center",
                        size=14)
            plt.figtext(right_center[0],
                        0.98,
                        "---- Virtual (" + EXP + ") ----",
                        va="center",
                        ha="center",
                        size=14)

        # plt.show(block=True)
        plt.savefig(fname=pjoin(
            root_dir, 'Figures',
            name + '_VerticalPlane_' + EXP.upper() + '.pdf'),
            bbox_inches='tight')


def plotHemisphereMap(titles,
                      final_dict_names,
                      mean_azi_ele_data,
                      confusion_data,
                      time_data,
                      coord_x,
                      coord_y,
                      all_colors,
                      EXP,
                      root_dir,
                      main_title=True,
                      sub_titles=True,
                      data_to_plot='Localization'):
    """ 
    Plot hemisphere maps of different experiment data.
    data_to_plot (string): 'Localization', 'ConfusionRate', or 'ResponseTime'.
    """

    # Plot localization plane data
    NUM_CHANNELS = 25
    plot_idcs = np.arange(NUM_CHANNELS)
    channels = range(0, NUM_CHANNELS)
    num_rows = 1
    num_cols = 4
    ring_color = 'grey'
    figsize = 3
    offs = 0.04
    r = 0.8

    DRAW_CHANNEL_NUMBER = False
    NORM_RATE = 50  # Confusion Norm Constant
    NORM_SEC = 10.0  # Time Norm Constant

    if data_to_plot == 'ResponseTime':
        cmap = cm.get_cmap('afmhot_r')
    if data_to_plot == 'ConfusionRate':
        cmap = cm.get_cmap('Reds')

    idx = 0
    fig, axs = plt.subplots(nrows=num_rows,
                            ncols=num_cols,
                            figsize=(4 * figsize, 1 * figsize),
                            gridspec_kw={
                                'hspace': 0.1,
                                'wspace': 0.05
                            })

    offset_idcs = np.concatenate((plot_idcs, plot_idcs + 25))
    final_idcs = [plot_idcs, plot_idcs, offset_idcs, offset_idcs, offset_idcs]

    plot_colors = [all_colors[i] for i in plot_idcs]
    # plot_colors_rep = plot_colors + plot_colors

    # ele_cls = [plot_colors, plot_colors_rep, plot_colors_rep, plot_colors_rep]

    markeredge_colors = plot_colors

    if EXP == 'Dynamic':
        conditions = [
            final_dict_names[0], final_dict_names[1], final_dict_names[3],
            final_dict_names[2]
        ]
    if EXP == 'Static':
        conditions = final_dict_names[1:]

    ext = []
    for col, condition in zip(range(num_cols), conditions):
        # L1 ring
        phi = np.linspace(0, 2 * np.pi, 100)
        axs[col].plot(r * np.cos(phi),
                      r * np.sin(phi),
                      alpha=0.3,
                      color=ring_color,
                      zorder=1)
        # L2 ring
        axs[col].plot(r * np.cos(phi) * 2 / 3,
                      r * np.sin(phi) * 2 / 3,
                      alpha=0.3,
                      color=ring_color,
                      zorder=1)
        # L3 ring
        axs[col].plot(r * np.cos(phi) * 1 / 3,
                      r * np.sin(phi) * 1 / 3,
                      alpha=0.3,
                      color=ring_color,
                      zorder=1)

        if data_to_plot == 'ConfusionRate':
            axs[col].scatter(r * coord_x,
                             r * coord_y,
                             facecolors=cmap(confusion_data[condition] * 100.0 /
                                             NORM_RATE),
                             edgecolors='k',
                             s=100,
                             alpha=1.0,
                             zorder=2)
        if data_to_plot == 'ResponseTime':
            if col == 0 and EXP == 'Static':
                idx += 1
                ext.append([
                    axs[col].get_window_extent(
                    ).x0, axs[col].get_window_extent().width
                ])
                axs[col].set_ylim(-1, 1)
                axs[col].set_xlim(-1, 1)
                axs[col].set_yticks([])
                axs[col].set_xticks([])
                continue
            else:
                val = (
                    np.median(time_data[condition], axis=0) - 3.0) / NORM_SEC
                axs[col].scatter(
                    r * coord_x,
                    r * coord_y,
                    facecolors=cmap(val),
                    edgecolors='k',
                    s=100,
                    alpha=1.0,
                    zorder=2)

        if data_to_plot == 'Localization':
            axs[col].scatter(r * coord_x,
                             r * coord_y,
                             facecolors=markeredge_colors,
                             edgecolors='k',
                             s=100,
                             alpha=0.6,
                             zorder=2)

            mkr_cls = plot_colors  # rand_cls[0]
            azi = mean_azi_ele_data[condition][plot_idcs, 0]
            ele = mean_azi_ele_data[condition][plot_idcs, 1]
            loc_data_x, loc_data_y = convertAziEleToPlane(azi, ele)
            axs[col].scatter(r * loc_data_x,
                             r * loc_data_y,
                             edgecolors='k',
                             s=30,
                             alpha=0.8,
                             zorder=2,
                             c=mkr_cls,
                             marker='D')
            for n in range(NUM_CHANNELS):
                axs[col].plot([r * coord_x[n], r * loc_data_x[n]], [r * coord_y[n],
                              r * loc_data_y[n]], color=mkr_cls[n], alpha=0.8, zorder=1, ls=':')

        if DRAW_CHANNEL_NUMBER:
            for channel in channels:
                axs[col].text(r * coord_x[channel] - offs,
                              r * coord_y[channel] - offs,
                              s=str(channel + 1),
                              zorder=3,
                              fontsize=6,
                              color='k')
        axs[col].set_ylim(-1, 1)
        axs[col].set_xlim(-1, 1)
        axs[col].set_yticks([])
        axs[col].set_xticks([])
        if sub_titles:  # HRTF method titles
            axs[col].set_title(titles[col])
        axs[col].set_aspect('equal')
        idx += 1
        ext.append([
            axs[col].get_window_extent().x0, axs[col].get_window_extent().width
        ])

    # from the axes bounding boxes calculate the optimal position of the column spanning title
    inv = fig.transFigure.inverted()
    width_left = ext[0][0] + (ext[1][0] + ext[1][1] - ext[0][0]) / 2.
    left_center = inv.transform((width_left, 1))
    width_right = ext[2][0] + (ext[3][0] + ext[3][1] - ext[2][0]) / 2.
    right_center = inv.transform((width_right, 1))

    if main_title:  # Real or Virtual titles
        plt.figtext(left_center[0],
                    0.98,
                    "---- Real (" + EXP + ") ----",
                    va="center",
                    ha="center",
                    size=14)
        plt.figtext(right_center[0],
                    0.98,
                    "---- Virtual (" + EXP + ") ----",
                    va="center",
                    ha="center",
                    size=14)

    if data_to_plot == 'ConfusionRate':
        color_bar_ax = fig.add_axes([0.41, 0.05, 0.2, 0.05])
        norm = colors.Normalize(vmin=0, vmax=NORM_RATE, clip=True)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=color_bar_ax,
                            orientation="horizontal")
        cbar.set_label('Quadrant Errors (%)', rotation=0)
    if data_to_plot == 'ResponseTime':
        color_bar_ax = fig.add_axes([0.41, 0.05, 0.2, 0.05])
        norm = colors.Normalize(vmin=0, vmax=NORM_SEC, clip=True)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=color_bar_ax,
                            orientation="horizontal")
        cbar.set_label('Median Response Time (s)', rotation=0)

    if data_to_plot == 'ConfusionRate':
        savename = 'SingleDirectionResults_Confusion_' + EXP.upper() + '.pdf'
    if data_to_plot == 'Localization':
        savename = 'SingleDirectionResults_Localization_' + EXP.upper(
        ) + '.pdf'
    if data_to_plot == 'ResponseTime':
        savename = 'SingleDirectionResults_ResponseTime_' + EXP.upper(
        ) + '.pdf'

    plt.savefig(fname=pjoin(root_dir, 'Figures', savename),
                bbox_inches='tight')
    plt.show(block=True)
    return


def plotResponseTimesQuantitative(time_data, EXP, real_dict_names, final_dict_names, xticklabels, coord_x, coord_y, root_dir):
    class median_conf_int:
        def __init__(self):
            self.low = 0
            self.high = 0
    if EXP == 'Static':
        conditions = final_dict_names[1:]
    if EXP == 'Dynamic':
        conditions = [
            final_dict_names[0], final_dict_names[1], final_dict_names[3],
            final_dict_names[2]
        ]

    size = 3
    median_idcs = np.array([24, 0, 23, 8, 16, 18, 20, 12, 4])
    dir_sets = [np.arange(25), np.arange(8), np.array(
        [1, 2, 9, 10]), np.array([6, 7, 14, 15, 22, 21])]

    fig, axs = plt.subplots(
        nrows=1, ncols=len(dir_sets), figsize=(len(dir_sets)*size, 1*size), sharey=False)
    x_offsets = [1, 2, 3, 4]

    for dirs, col in zip(dir_sets, range(len(dir_sets))):

        POOL_DIR_DATA = False
        SEC_OFF = 3.0

        if POOL_DIR_DATA:
            data = np.zeros((len(conditions), 16*dirs.size))
            for condition, idx in zip(conditions, range(len(conditions))):
                data[idx, :] = (time_data[condition]
                                [:, dirs] - SEC_OFF).flatten()
        else:
            data = np.zeros((len(conditions), 16))
            for condition, idx in zip(conditions, range(len(conditions))):
                data[idx, :] = np.median(
                    time_data[condition][:, dirs] - SEC_OFF, axis=1)
        pairs_to_be_tested = [[1, 2], [1, 3]]
        pvals, effect_sizes = posthoc_wilcoxon(
            data, pairs_to_be_tested, alternative_h='two-sided', p_adjust='BH')
        for dict_name, idx, x_off in zip(conditions, range(len(conditions)), x_offsets):
            if EXP == 'Static' and dict_name == 'StaticOpenEars':
                continue
            dir_mean_resp_times = data[idx, :]
            median_time = np.median(dir_mean_resp_times)  # participant median
            median_int = median_conf_int()
            median_int.low = np.quantile(dir_mean_resp_times, q=0.25)
            median_int.high = np.quantile(dir_mean_resp_times, q=0.75)
            asymmetric_IQR = np.array([median_time - median_int.low,
                                       median_int.high - median_time])[:, None]
            axs[col].errorbar(x_off, median_time, capsize=2.0, linestyle='none',
                              xerr=0, yerr=asymmetric_IQR, color='k', zorder=2)
            axs[col].plot(x_off, median_time, 's', markersize=6, markerfacecolor='white',
                          markeredgecolor='k', zorder=3)

        y_refs = [2, 1.5]
        for i, y_ref in zip(range(len(pairs_to_be_tested)), y_refs):
            # if not dict_name == 'DynamicOpenEars':
            if dict_name not in real_dict_names:
                if pvals[i] >= 0.05:
                    pval_str = 'ns'
                    xoffs = -0.125
                    yoffs = -0.4
                if pvals[i] < 0.05:
                    pval_str = '*'
                    xoffs = -0.075
                    yoffs = -0.5
                if pvals[i] < 0.01:
                    pval_str = '**'
                    xoffs = -0.15
                    yoffs = -0.5
                if pvals[i] < 0.001:
                    pval_str = '***'
                    xoffs = -0.175
                    yoffs = -0.5
                brace_start = x_offsets[pairs_to_be_tested[i][0]] + 0.1
                brace_end = x_offsets[pairs_to_be_tested[i][1]] - 0.1

                axs[col].plot([brace_start, brace_end],
                              [y_ref, y_ref],
                              color='k')
                axs[col].plot([brace_start, brace_start],
                              [y_ref, y_ref+0.1],
                              color='k')
                axs[col].plot([brace_end, brace_end],
                              [y_ref, y_ref+0.1],
                              color='k')
                axs[col].text((brace_start + brace_end) / 2 + xoffs,
                              y_ref + yoffs, s=pval_str, fontsize=10)
                # eff_str = ''
                if np.abs(effect_sizes[i]) >= 0.75:
                    eff_str = r'$\Delta$'
                    xoffs = -0.075
                    yoffs = 0.1
                    axs[col].text((brace_start + brace_end) / 2 + xoffs,
                                  y_ref + yoffs, s=eff_str, fontsize=10)

        axs[col].set_xlim(0, 5)
        if col == 0:
            axs[col].set_ylabel('Response time (s)')
        # plt.xlabel('Condition')
        axs[col].set_xticks(x_offsets, xticklabels)
        # axs[col].set_yticks(np.arange(0, 11), ['0', '', '2',
        #                    '', '4', '', '6', '', '8', '', '10'])
        axs[col].set_yticks(np.arange(1, 12), ['1', '', '3',
                            '', '5', '', '7', '', '9', '', ''])
        # axs[col].set_yticks(np.arange(0, 11))
        axs[col].set_ylim(1, 11)
        axs[col].grid(axis='y')

        if EXP == 'Dynamic':
            pos_size_list = [0.015, 0.61, 0.4, 0.4]
        else:
            pos_size_list = [0.015, 0.1, 0.4, 0.4]

        renderInsetAxis(axs[col], dirs, coord_x,
                        coord_y, pos_size_list, mkr_size=15)
    # plt.show(block=True)
    plt.savefig(fname=pjoin(
        root_dir, 'Figures', EXP + '_ResponseTimes_Quantitative.pdf'),
        bbox_inches='tight')
    print('')
