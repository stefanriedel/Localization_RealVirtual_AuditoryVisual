import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from os.path import dirname, join as pjoin
from matplotlib import cm
import matplotlib.colors as colors
from Utility.pairwiseTests import posthoc_wilcoxon, posthoc_ttest
from scipy.spatial.distance import cdist, euclidean
import warnings
import scipy.stats as stats

from Utility.wilcoxon import stats_wilcoxon
from Utility.mannwhitneyu import stats_mannwhitneyu

from Utility.loudspeakerPositions import azi, ele


# Render vertical plane plots
name_list = [
    '0DEG', '-30DEG', '30DEG', '-90DEG', '90DEG', '180DEG', '-150DEG', '150DEG'
]
name_list_jasa_static = [
    'Fig6a', 'Fig4b', 'Fig4a', 'Fig5b', 'Fig5a', 'Fig6b', '-150DEG', '150DEG'
]
name_list_jasa_dynamic = [
    'Fig6c', 'Fig4d', 'Fig4c', 'Fig5d', 'Fig5c', 'Fig6d', '-150DEG', '150DEG'
]
name_list_azi = [
    '0DEG', '30DEG', '60DEG'
]

deg_list = [
    r'$0$' + '°', r'$-30$' + '°', r'$30$' + '°', r'$-90$' + '°', r'$90$' + '°',
    r'$180$' + '°', r'$-150$' + '°', r'$150$' + '°'
]
deg_list_azi = [
    r'$0$' + '°', r'$30$' + '°', r'$60$' + '°'
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
idcs_list_azi = [
    np.array([5, 4, 3, 2, 1, 8, 7, 6]) - 1,  # 0 DEG ELE
    np.array([13, 12, 11, 10, 9, 16, 15, 14]) - 1,  # 30 DEG ELE,
    np.array([19, 18, 17, 20]) - 1,  # 60 DEG ELE
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
pairtest_list_azi = [
    [[0, 1], [1, 2]],
    [[0, 1], [1, 2]],
    [[0, 1], [1, 2]]
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
target_azi_list = [
    np.array([180, 150, 90, 30, 0, -30, -90, -150]),
    np.array([180, 150, 90, 30, 0, -30, -90, -150]),
    np.array([180, 90, 0, -90])
    #np.array([30, 0, -30]),
]

target_distance_hits = [
    np.array([np.array([-7.5,7.5]), np.array([-7.5,7.5]), np.array([-7.5,7.5]), np.array([-7.5,15]), np.array([-15,15])]),
    np.array([np.array([-7.5,7.5]), np.array([-7.5,7.5]), np.array([-7.5,15])]),
    np.array([np.array([-7.5, 15]), np.array([-15,15])]),
    np.array([np.array([-7.5,7.5]), np.array([-7.5,7.5]), np.array([-7.5,15]), np.array([-15,15])]),
    np.array([np.array([-7.5, 15]), np.array([-15,15]), np.array([-15,15])]),
    np.array([np.array([-7.5, 15]), np.array([-15,15]), np.array([-15,15])]),
    np.array([np.array([-7.5,15]), np.array([-15,15])]),
    np.array([np.array([-7.5,15]), np.array([-15,15])]),
]

target_distance_hits_azi = [
    np.array([np.array([-15,15]), np.array([-30,15]), np.array([-30,30]), np.array([-15,30]), np.array([-15,15]),  np.array([-30,15]),  np.array([-30,30]), np.array([-15,30])]),
    np.array([np.array([-15,15]), np.array([-30,15]), np.array([-30,30]), np.array([-15,30]), np.array([-15,15]),  np.array([-30,15]),  np.array([-30,30]), np.array([-15,30])]),
    np.array([np.array([-45, 45]), np.array([-45, 45]), np.array([-45, 45]), np.array([-45, 45])]),
]


SEC_OFF = 3.0
NORM_SEC = 10.0

# For error metric table
NUM_CHANNELS = 25
L1_idcs = np.arange(8)
L2_idcs = np.arange(8, 16)
L3_idcs = np.arange(16, 20)
dir_sets = [
    L1_idcs,
    np.array([0]),
    np.concatenate((L2_idcs, L3_idcs)),
    np.array([4]),
    np.arange(25)
]
dirset_names = ['Horizontal', 'Frontal', 'Elevated', 'Rear', 'Overall']

# 0 DEG # -30 DEG # 30 DEG # -90 DEG# 90 DEG # 180 DEG # -150 DEG # 150 DEG

# title_bool_list = [True, False, True, False, True, False, False, True]
title_bool_list = [True, False, True, False, True, False, True, True]
title_bool_list_azi = [False, True, True]


xaxis_bool_list = [False, True, False, True, False, True, True, True]
xaxis_bool_list_azi = [True, True, True]


all_colors = ['tab:orange'] * 8 + ['tab:red'] * 8 + ['tab:purple'] * 4 + [
    'tab:brown'
] + ['tab:green'] * 3 + ['tab:blue']

all_colors_hex_alpha = ['#ffbf86'] * 8 + ['#ea9393'] * 8 + ['#c9b3de'] * 4 + ['#c5aaa5']  + ['#95cf95'] * 3 + ['#8fbbd9']

def MAD(data, axis=0):
        return np.nanmedian(np.abs(data - np.nanmedian(data, axis=axis)), axis=axis)


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


def computeCorrelationConfusionRateResponseTimes(local_azi_ele_data, time_data, final_dict_names):
    # all directions
    dirs = np.arange(25)

    #subset 
    m30 = np.array([8, 23, 16]) - 1  # -30 DEG
    p30 = np.array([2, 10]) - 1  # 30 DEG
    m90 = np.array([7, 22, 15]) - 1  # -90 DEG
    p90 = np.array([3, 11]) - 1  # 90 DEG
    #p0 = np.array([25, 1, 24, 9, 17]) - 1 # 0 DEG
    p0 = np.array([1, 24, 9]) - 1 # 0 DEG
    dirs = np.hstack((m30, p30, m90, p90, p0), dtype=int)
    target_elevations = ele

    target_distances = np.zeros((NUM_CHANNELS, 2))
    for vertical_idcs, target_distances_ele in zip(idcs_list, target_distance_hits):
        for tr, idx in zip(vertical_idcs, range(len(vertical_idcs))):
            target_distances[tr, 0] = target_distances_ele[idx, 0]
            target_distances[tr, 1] = target_distances_ele[idx, 1]

    # Zenith loudspeaker
    target_distances[20, 0] = -15.0
    target_distances[20, 1] = 0.0

    confusion_rate = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }
    
    median_response_times = {
            dict_name: np.array([])
            for dict_name in final_dict_names
        }

    for dict_name in final_dict_names:
        num_trials = len(local_azi_ele_data[dict_name])

        local_confusion_rates = np.zeros(NUM_CHANNELS)
        for tr in range(NUM_CHANNELS):
            if num_trials == NUM_CHANNELS:
                responses_azi_ele = local_azi_ele_data[dict_name][tr]
            else:
                first_resp = local_azi_ele_data[dict_name][tr]
                doubled_resp = local_azi_ele_data[dict_name][tr +
                                                                NUM_CHANNELS]
                responses_azi_ele = np.concatenate(
                    (first_resp, doubled_resp))

            elevation_ratings = responses_azi_ele[:, 1]
            elevation_ratings = elevation_ratings[~np.isnan(elevation_ratings)]
            elevation_distances = elevation_ratings - target_elevations[tr]
            confusions = np.logical_or(elevation_distances < target_distances[tr,0],  elevation_distances > target_distances[tr,1])
            CR = float(confusions.sum()) / float(confusions.size)
            local_confusion_rates[tr] = CR


        median_response_times[dict_name] = np.median(time_data[dict_name], axis=0)
        confusion_rate[dict_name] = local_confusion_rates


    idcs_eval = dirs
    rt = np.array([])
    cr = np.array([])

    for dict_name in final_dict_names:
        rt = np.hstack((rt, median_response_times[dict_name][idcs_eval]))
        cr = np.hstack((cr, confusion_rate[dict_name][idcs_eval]))
    res = stats.pearsonr(cr * 100.0, rt)

    res_linreg = stats.linregress(cr * 100.0, rt)
    slope = res_linreg.slope
    intercept = res_linreg.intercept
    x= np.linspace(0,100,100)
    y = x * slope + intercept
    
    fs = 10
    scale = 2
    plt.figure(figsize=(2*scale,2*scale))
    plt.plot(x, y, color='grey', zorder=0)
    plt.scatter(cr * 100.0, rt, zorder=1)
    if res_linreg.pvalue < 0.001:
        plt.text(70, 5, s = 'p < 0.001', fontsize=fs)
    else:
        plt.text(70, 5, s = 'p = ' + str(round(res_linreg.pvalue, 3)), fontsize=fs)

    plt.text(70, 4, s = 'r = ' + str(round(res_linreg.rvalue, 2)), fontsize=fs)
    plt.xlim(0,100)
    plt.ylim(2,14)
    plt.grid()
    plt.ylabel('Median Response Time (sec.)')
    plt.xlabel('Local Confusion Rate (%)')
    plt.savefig('Figures/Correlation_CR_RT.png')

    return


def computeAndSaveErrorMetrics(save_dir, EXP, dir_sets, dirset_names,
                               final_dict_names, local_azi_ele_data,
                               targets_azi_ele, QE_data, ABS_BIAS):
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
        quadrant_error = {
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

                azi_error_cand = responses_azi_ele[:, 0] - targets_azi_ele[tr % NUM_CHANNELS,
                                                                           0]
                azi_error_cand[azi_error_cand > 180.0] = 360.0 - \
                    azi_error_cand[azi_error_cand > 180.0]
                azi_error_cand[azi_error_cand < -180.0] = 360.0 + \
                    azi_error_cand[azi_error_cand < -180.0]

                ele_error_cand = responses_azi_ele[:, 1] - \
                    targets_azi_ele[tr % NUM_CHANNELS, 1]

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
                if ABS_BIAS:
                    azi_bias = np.abs(np.nanmean(azi_errors, axis=0))
                    ele_bias = np.abs(np.nanmean(ele_errors, axis=0))
                else:
                    azi_bias = np.nanmean(azi_errors, axis=0)
                    ele_bias = np.nanmean(ele_errors, axis=0)
            local_azi_error_rms[dict_name] = azi_error_rms
            local_ele_error_rms[dict_name] = ele_error_rms
            local_azi_bias[dict_name] = azi_bias
            local_ele_bias[dict_name] = ele_bias
            quadrant_error[dict_name] = np.mean(QE_data[dict_name][dirs])

        # Save to disk with dirset name and dictionary name
        metric_data_list = [
            local_azi_error_rms, local_azi_bias, local_ele_error_rms,
            local_ele_bias, quadrant_error
        ]
        metric_name_list = [
            'AzimuthError', 'AzimuthBias', 'ElevationError', 'ElevationBias', 'QuadrantError'
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
                        pval, effect_size, T, z = posthoc_wilcoxon(
                            data,
                            pairs,
                            method='approx',
                            alternative_h='two-sided',
                            p_adjust=None)
                        significant = pval < 0.05
                        strong_effect = np.abs(effect_size) >= 0.75

                    if significant and ~strong_effect and dict_name != ref_name:
                        med_mad_string = '\\cellcolor{lightgray!50} ' "%.1f" % np.nanmedian(metric_data) + ' $\pm$ ' + "%.1f" % MAD(metric_data) + ' \\,'
                    elif significant and strong_effect and dict_name != ref_name:
                        med_mad_string = '\\cellcolor{lightgray!50} ' + '\\textbf{' + "%.1f" % np.nanmedian(metric_data) + ' $\pm$ ' + "%.1f" % MAD(metric_data) + '}' + ' \\,'
                    else:
                        med_mad_string = "%.1f" % np.nanmedian(metric_data) + ' $\pm$ ' + "%.1f" % MAD(metric_data)

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


def computeMetrics(response_angles,
                   target_angles,
                   median_statistic=True):
    N = response_angles.shape[1]
    if not median_statistic:
        response_angles = response_angles.flatten()
        target_angles = np.tile(target_angles,
                                    reps=(N, 1)).flatten('F')
        non_nan_idcs = ~np.isnan(response_angles)

        res = stats.linregress(target_angles[non_nan_idcs],
                               response_angles[non_nan_idcs])
        slope = res.slope
        intercept = res.intercept

        sigma = np.sqrt(
            np.mean((target_angles[non_nan_idcs] -
                     response_angles[non_nan_idcs])**2))

        bias = np.mean((response_angles[non_nan_idcs] -
                        target_angles[non_nan_idcs]))
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
            resp_elevations = response_angles[:, n]
            non_nan_idcs = ~np.isnan(resp_elevations)

            if not target_angles[non_nan_idcs].size >= 1:
                continue

            res = stats.linregress(target_angles[non_nan_idcs],
                                   resp_elevations[non_nan_idcs])
            slope[n] = res.slope
            intercept[n] = res.intercept

            sigma[n] = np.sqrt(
                np.mean((target_angles[non_nan_idcs] -
                         resp_elevations[non_nan_idcs])**2))

            bias[n] = np.mean((resp_elevations[non_nan_idcs] -
                               target_angles[non_nan_idcs]))
        slope = np.nanmedian(slope)
        intercept = np.nanmedian(intercept)
        bias = np.nanmedian(bias)
        sigma = np.nanmedian(sigma)
    return slope, intercept, sigma, bias


def renderInsetAxis(ax,
                    active_idcs,
                    coord_x,
                    coord_y,
                    pos_size_list,
                    mkr_size=15):
    # inset axes for pictogram of source directions
    ring_color = 'lightgrey'
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
               #alpha=0.3,
               color=ring_color,
               zorder=5)
    # L2 ring
    axins.plot(r * np.cos(phi) * 2 / 3,
               r * np.sin(phi) * 2 / 3,
               #alpha=0.3,
               color=ring_color,
               zorder=5)
    # L3 ring
    axins.plot(r * np.cos(phi) * 1 / 3,
               r * np.sin(phi) * 1 / 3,
               #alpha=0.3,
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
                       deg_list, title_bool_list, titles, xaxis_bool_list,
                       final_dict_names, local_azi_ele_data, coord_x, coord_y,
                       all_colors, EXP, root_dir, plot_avg_ele, render_with_jasa_names):
    for mvp_idcs, pairs_to_be_tested, target_elevations, name, deg, title_bool, xaxis_bool, target_distances in zip(
            idcs_list, pairtest_list, target_ele_list, name_list, deg_list,
            title_bool_list, xaxis_bool_list, target_distance_hits):
        num_rows = 1
        num_cols = 4
        figsize = 3
        fig, axs = plt.subplots(nrows=num_rows,
                                ncols=num_cols,
                                sharey=True,
                                figsize=(4 * figsize, 1.1 * figsize),
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
            conditions_ele_wilcox = conditions_ele

            # Case of repeated responses
            if col >= 1:
                cond_ele_first = np.asarray(
                    local_azi_ele_data[condition])[mvp_idcs, :, 1]
                cond_ele_second = np.asarray(
                    local_azi_ele_data[condition])[mvp_idcs + 25, :, 1]
                stacked = np.array([cond_ele_first, cond_ele_second])
                conditions_ele = np.nanmean(stacked, axis=0)

                USE_FIRST_RESPONSE_WILCOX = False
                if USE_FIRST_RESPONSE_WILCOX:
                    conditions_ele_wilcox = cond_ele_first
                else:
                    conditions_ele_wilcox = conditions_ele

            slope, intercept, sigma, bias = computeMetrics(
                conditions_ele, target_elevations, median_statistic=True)
            
            g_x = 44
            axs[col].text(x=g_x,
                          y=3,
                          s=r'$g = $' + str(round(slope, 2)),
                          fontsize=10)
            x = np.linspace(-32.5, 90, 100)
            axs[col].plot(x,
                          slope * x + intercept,
                          zorder=0,
                          color='gray',
                          ls=':')

            data = conditions_ele_wilcox
            pvals, effect_sizes, T, z = posthoc_wilcoxon(data,
                                                   pairs_to_be_tested,
                                                   alternative_h='two-sided',
                                                   p_adjust=None)
            
            # pvals, t = posthoc_ttest(data,
            #                         pairs_to_be_tested,
            #                         alternative_h='two-sided',
            #                         p_adjust=None)
            

            significant = pvals < 0.05

            if col >= 1:
                confusions_per_condition = np.zeros((mvp_idcs.size, 32), dtype=bool)
            else:
                confusions_per_condition = np.zeros((mvp_idcs.size, 16), dtype=bool)

            confusion_rates = np.zeros(mvp_idcs.size)
            for i in range(mvp_idcs.size):
                elevation_ratings = local_azi_ele_data[condition][
                    mvp_idcs[i]][:, 1]
                if col >= 1:
                    elevation_ratings = np.concatenate(
                        (elevation_ratings,
                         local_azi_ele_data[condition][mvp_idcs[i] + 25][:,
                                                                         1]))
                if plot_avg_ele: # PLOT MEAN DATA OF DOUBLED RESPONSES
                    elevation_ratings = conditions_ele[i, :]
                median_ratings[i] = np.nanmedian(elevation_ratings)
                axs[col].scatter(
                    target_elevations[i] +
                    (np.random.rand(elevation_ratings.size) - 0.5) * 7.5,
                    elevation_ratings,
                    edgecolors='k',
                    s=15,
                    zorder=2,
                    c=all_colors[mvp_idcs[i]])
                
                # Consider only local responses (quadrant errors are nans)
                elevation_ratings = elevation_ratings[~np.isnan(elevation_ratings)]

                elevation_distances = elevation_ratings - target_elevations[i]

                confusions = np.logical_or(elevation_distances < target_distances[i,0],  elevation_distances > target_distances[i,1])

                confusion_rate = float(confusions.sum()) / float(confusions.size)

                confusion_rates[i] = confusion_rate

                #add grid lines manually 
                if i == 0:
                    axs[col].text(-44, -42, s='LCR:', fontsize=9, style='italic')
                axs[col].text(target_elevations[i] - 2, -42, s=str(round(confusion_rate*100)) , fontsize=9, style='italic')
                if i == mvp_idcs.size - 1:
                    axs[col].text(target_elevations[i] + 8, -42, s=' (%)' , fontsize=9, style='italic')
      
                grid_elevations = np.array([-15,0,15,30,60])
                for i in range(grid_elevations.size):
                    # horizontal line
                    axs[col].plot([-45, 90], [grid_elevations[i], grid_elevations[i]], zorder=0, lw=0.5, color='gray', ls=(0, (1, 3)))
                    # vertical line
                    axs[col].plot([grid_elevations[i], grid_elevations[i]], [-35, 90], zorder=0, lw=0.5, color='gray', ls=(0, (1, 3)))
                

            axs[col].text(x=g_x,
                    y=-13,
                    s=r'$\overline{\mathrm{LCR}} = $' + str(round(np.mean(confusion_rates*100))) + '%',
                    fontsize=10)

            PLOT_PVALUES = False

            if PLOT_PVALUES:
                for i in range(len(pairs_to_be_tested)):
                    if True:  # significant[i] == True:
                        if pvals[i] >= 0.05:
                            pval_str = 'ns'
                            offs = -2
                            voff = 0
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
                            offs = -3.5
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

                        # if np.abs(effect_sizes[i]) >= 0.75:
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

            axs[col].plot([-32.5, 90], [-32.5, 90], color='grey', zorder=1)
            axs[col].scatter(target_elevations,
                             median_ratings,
                             marker='D',
                             edgecolors='k',
                             linewidth=1,
                             facecolors='white',
                             zorder=3,
                             s=30)
            


            if col == 0:
                axs[col].set_ylabel('Elevation Response (deg.)')
                renderInsetAxis(axs[col],
                        mvp_idcs,
                        coord_x,
                        coord_y,
                        [0.02, 0.585, 0.38, 0.38],
                        #[0.005, 0.6, 0.38, 0.38],
                        mkr_size=20)
            axs[col].set_xlim(-45, 90)
            axs[col].set_ylim(-45, 90)
            if xaxis_bool:
                axs[col].set_xticks([-15, 0, 15, 30, 60])
                axs[col].set_xlabel('Elevation Target (deg.)')
            else:
                axs[col].set_xticks([-15, 0, 15, 30, 60])
                #axs[col].set_xticklabels([])

            axs[col].set_yticks([-15, 0, 15, 30, 60])
            if title_bool:
                axs[col].set_title(titles[col])
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

        #plt.show(block=True)
        if render_with_jasa_names:
            plt.savefig(fname=pjoin(
                root_dir, 'Figures',
                name + '.eps'), bbox_inches='tight', dpi=300)
        else:
            plt.savefig(fname=pjoin(
                root_dir, 'Figures',
                name + '_VerticalPlane_' + EXP.upper() + '.png'), bbox_inches='tight', dpi=300)


def computeSlopeData(final_dict_names, local_azi_ele_data, EXP, DIM, root_dir):
    if EXP == 'Static':
        conditions = final_dict_names[1:]
    if EXP == 'Dynamic':
        conditions = [
            final_dict_names[0], final_dict_names[1], final_dict_names[3],
            final_dict_names[2]
        ]

    if DIM == 'Azimuth':
        group_idcs_list = idcs_list_azi
        group_idcs_list.append(np.concatenate((idcs_list_azi[0], idcs_list_azi[1], idcs_list_azi[2])))
        group_name_list = name_list_azi
        group_name_list.append('0DEG+30DEG+60DEG')
        target_list = target_azi_list
        target_list.append(np.concatenate((target_list[0], target_list[1], target_list[2])))
        angle_idx = 0
    elif DIM == 'Elevation':
        group_idcs_list = idcs_list
        group_name_list = name_list
        target_list = target_ele_list
        angle_idx = 1
    else:
        print('Dimension not properply specified!')

    regression_slope = {
            dict_name: {}
            for dict_name in conditions
        }
    for dict_name in conditions:
        for idcs, name, target_angles  in zip(group_idcs_list, group_name_list, target_list):
            if dict_name == 'StaticOpenEars' or dict_name == 'DynamicOpenEars':
                # single responses
                response_angles = np.asarray(local_azi_ele_data[dict_name])[idcs, :, angle_idx]
            else:
                # averaging of doubled responses before regression
                response_angles_first = np.asarray(
                    local_azi_ele_data[dict_name])[idcs, :, angle_idx]
                response_angles_second = np.asarray(
                    local_azi_ele_data[dict_name])[idcs + 25, :, angle_idx]
                stacked = np.array([response_angles_first, response_angles_second])
                response_angles = np.nanmean(stacked, axis=0)

            if DIM == 'Azimuth':
                response_angles[response_angles < -170.0] = (response_angles[response_angles < -170.0] + 360.0) % 360.0

            N = response_angles.shape[1]
            slope = np.zeros(N)
            slope[:] = np.nan

            for n in range(N):
                resp_n = response_angles[:, n]
                non_nan_idcs = ~np.isnan(resp_n)

                if not target_angles[non_nan_idcs].size >= 1:
                    continue

                res = stats.linregress(target_angles[non_nan_idcs],
                                        resp_n[non_nan_idcs])
                slope[n] = res.slope
            regression_slope[dict_name][name] = slope 

    if DIM == 'Elevation':
        savename  = 'SlopeDataElevation'
    if DIM == 'Azimuth':
        savename  = 'SlopeDataAzimuth'
    np.save(file=pjoin(root_dir,
                'ErrorMetricData', savename + EXP + '.npy'),
                arr=regression_slope,
                allow_pickle=True)   
    return

def testGroupedSlopeData(first_condition_data, second_condition_data, condition_pair, planes, PAIRED_SAMPLES=True, SUBJ_IDCS=[np.arange(16), np.arange(16)]):
    g_first = np.array([])
    g_second = np.array([])

    for plane, i in zip(planes, range(len(planes))):
        first_data = first_condition_data[condition_pair[0]][plane]
        second_data = second_condition_data[condition_pair[1]][plane]

        first_data = first_data[SUBJ_IDCS[0]]
        second_data = second_data[SUBJ_IDCS[1]]

        g_first = np.concatenate((g_first, first_data))
        g_second = np.concatenate((g_second, second_data))

    non_nan_idcs_first = ~np.isnan(g_first)
    non_nan_idcs_second = ~np.isnan(g_second)
    non_nan_idcs = np.logical_and(non_nan_idcs_first, non_nan_idcs_second)

    g_first = g_first[non_nan_idcs]
    g_second = g_second[non_nan_idcs]

    if 0:
        plt.figure()
        plt.boxplot([g_first, g_second])
        plt.show(block=True)

    #res = stats.ttest_rel(g_first, g_second, alternative='two-sided')
    #res = stats.wilcoxon(g_first, g_second, alternative='two-sided')
    res, T_plus, T_minus = stats_wilcoxon(g_first, g_second, alternative='two-sided')
    
    cohens_d = (np.mean(g_first) - np.mean(g_second)) / (np.sqrt((np.std(g_first) ** 2 + np.std(g_second) ** 2) / 2))

    print('Conditions: ' + str(condition_pair), 
          #' --> ' + str(np.mean(g_first)) + ' vs. ' + str(np.mean(g_second)),
          ' --> ' + str(np.round(np.median(g_first), 2)) + ' vs. ' + str(np.round(np.median(g_second), 2)),
          'pvalue: ' + str(res[1]),
          'T(' + str(g_first.size) + ') = ' + str(T_plus - T_minus))

          #'cohens_d:' + str(cohens_d))
    return

def testGroupedSlopeDataConstantSampleSize(first_condition_data, second_condition_data, condition_pair, planes, SUBJ_IDCS=[np.arange(16), np.arange(16)], PAIRED_SAMPLES=True, NONPARAM=True):
    g_first = np.zeros((len(SUBJ_IDCS[0]), len(planes)))
    g_second = np.zeros((len(SUBJ_IDCS[1]), len(planes)))

    for plane, i in zip(planes, range(len(planes))):
        first_data = first_condition_data[condition_pair[0]][plane]
        second_data = second_condition_data[condition_pair[1]][plane]

        first_data = first_data[SUBJ_IDCS[0]]
        second_data = second_data[SUBJ_IDCS[1]]

        g_first[:, i] = first_data
        g_second[:, i] = second_data

    g_first = np.nanmean(g_first, axis=1)
    g_second = np.nanmean(g_second, axis=1)

    non_nan_idcs_first = ~np.isnan(g_first)
    non_nan_idcs_second = ~np.isnan(g_second)
    non_nan_idcs = np.logical_and(non_nan_idcs_first, non_nan_idcs_second)

    g_first = g_first[non_nan_idcs]
    g_second = g_second[non_nan_idcs]

    if 0:
        plt.figure()
        plt.boxplot([g_first, g_second])
        plt.show(block=True)



    if PAIRED_SAMPLES:
        if NONPARAM:
            res, T_plus, T_minus = stats_wilcoxon(g_first, g_second, alternative='two-sided')
            print('Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.median(g_first), 2)) + ' vs. ' + str(np.round(np.median(g_second), 2)),
                'pvalue: ' + str(res[1]),
                'T(' + str(g_first.size) + ') = ' + str(T_plus - T_minus))
        else:
            res = stats.ttest_rel(g_first, g_second, alternative='two-sided')
            print('Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.mean(g_first), 2)) + ' vs. ' + str(np.round(np.mean(g_second), 2)),
                'pvalue: ' + str(res[1]),
                't(' + str(g_first.size) + ') = ' + str(res[0]))
    else:
        if NONPARAM:
            res = stats.mannwhitneyu(g_first, g_second, alternative='two-sided')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.median(g_first), 2)) + ' vs. ' + str(np.round(np.median(g_second), 2)),
                'pvalue: ' + str(res[1]), 'U1(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]))
        else:
            res = stats.ttest_ind(g_first, g_second, alternative='two-sided')
            print('Conditions: ' + str(condition_pair), 
                ' --> ' + str(np.round(np.mean(g_first), 2)) + ' vs. ' + str(np.round(np.mean(g_second), 2)),
                'pvalue: ' + str(res[1]),
                't(' + str(g_first.size) + ') = ' + str(res[0]))
    return  


def computeLocalConfusionData(final_dict_names, local_azi_ele_data, EXP, DIM, root_dir):
    if EXP == 'Static':
        conditions = final_dict_names[1:]
    if EXP == 'Dynamic':
        conditions = [
            final_dict_names[0], final_dict_names[1], final_dict_names[3],
            final_dict_names[2]
        ]

    if DIM == 'Elevation':
        target_angles = ele
        target_distances_ele_l1 = np.array([np.array([-7.5, 7.5]), np.array([-7.5, 15]), np.array([-7.5, 15]), np.array([-7.5, 15]), np.array([-7.5, 15]), np.array([-7.5, 15]), np.array([-7.5, 7.5]), np.array([-7.5, 7.5])])
        target_distances_ele_l2 = np.array([np.array([-7.5, 15]), np.array([-15, 15]), np.array([-15, 15]), np.array([-15, 15]), np.array([-15, 15]), np.array([-15, 15]), np.array([-7.5, 15]), np.array([-7.5, 15])])
        target_distances_ele_l3 = np.array([np.array([-15, 15]), np.array([-15, 15]), np.array([-15, 15]), np.array([-15, 15])])
        target_distances_rest = np.array([np.array([-15, 0]), np.array([-7.5, 7.5]), np.array([-7.5, 7.5]), np.array([-7.5, 7.5]), np.array([-7.5, 7.5])])
        target_distances =  np.concatenate((target_distances_ele_l1,target_distances_ele_l2,target_distances_ele_l3,target_distances_rest))
    if DIM == 'Azimuth':
        target_angles = azi
        target_distances_azi_l1 = np.array([np.array([-15,15]), np.array([-15, 30]), np.array([-30,30]), np.array([-30,15]), np.array([-15,15]),  np.array([-15,30]),  np.array([-30,30]), np.array([-30,15])])
        target_distances_azi_l2 = np.array([np.array([-15,15]), np.array([-15, 30]), np.array([-30,30]), np.array([-30,15]), np.array([-15,15]),  np.array([-15,30]),  np.array([-30,30]), np.array([-30,15])])
        target_distances_azi_l3 = np.array([np.array([-45, 45]), np.array([-45, 45]), np.array([-45, 45]), np.array([-45, 45])])
        target_distances_rest = np.array([np.array([-180, 180]), np.array([-30,30]), np.array([-30,15]), np.array([-15, 15]), np.array([-15, 15])])
        target_distances =  np.concatenate((target_distances_azi_l1, target_distances_azi_l2, target_distances_azi_l3, target_distances_rest))
    
    local_confusions = {
            dict_name: {}
            for dict_name in conditions
        }
    for dict_name in conditions:
        local_response_list = []
        confusions_list = []
        confusion_rate_list = []
        for tr in range(NUM_CHANNELS):
            if DIM == 'Elevation':
                angle_idx = 1
            if DIM == 'Azimuth':
                angle_idx = 0

            angular_ratings = local_azi_ele_data[dict_name][tr][:, angle_idx]
            if dict_name != 'StaticOpenEars' and dict_name != 'DynamicOpenEars' :
                angular_ratings = np.concatenate(
                    (angular_ratings,
                        local_azi_ele_data[dict_name][tr + 25][:, angle_idx]))
                
            # Init confusions array with nans
            num_raw_responses = angular_ratings.size
            confusions_with_nans = np.empty(num_raw_responses)
            confusions_with_nans[:] = np.nan      

            # Consider only local responses (quadrant errors are nans)
            local_response_idcs = ~np.isnan(angular_ratings)
            local_angular_ratings = angular_ratings[local_response_idcs]

            if DIM == 'Azimuth':
                # Special case for azimuth data at +-180 degree
                if target_angles[tr] == 180:
                    local_angular_ratings = (local_angular_ratings + 360.0) % 360.0 

            angular_distances = local_angular_ratings - target_angles[tr]
            confusions = np.logical_or(angular_distances < target_distances[tr,0],  angular_distances > target_distances[tr,1])

            for local_idx, idx in zip(np.where(local_response_idcs == True)[0], range(num_raw_responses)):
                confusions_with_nans[local_idx] = confusions[idx]

            confusion_rate = float(np.nansum(confusions_with_nans)) / float(confusions.size)

            local_response_list.append(local_response_idcs)
            confusions_list.append(confusions_with_nans)
            confusion_rate_list.append(confusion_rate)

        local_confusions[dict_name] = {'LocalResponseIdcs': local_response_list, 'Confusions': confusions_list, 'ConfusionRates': confusion_rate_list}

    if DIM == 'Elevation':
        savename  = 'LocalConfusionDataElevation'
    if DIM == 'Azimuth':
        savename  = 'LocalConfusionDataAzimuth'
    np.save(file=pjoin(root_dir,
                'ErrorMetricData', savename + EXP + '.npy'),
                arr=local_confusions,
                allow_pickle=True)
    return

def testGroupedLocalConfusionRate(first_condition_data, second_condition_data, condition_pair, direction_pair, SUBJ_IDCS=[np.arange(16), np.arange(16)]):
    # Only test over equal numbers of directions
    assert(len(direction_pair[0]) == len(direction_pair[1]))

    N_first = np.zeros(len(direction_pair[0]))
    N_second = np.zeros(len(direction_pair[1]))
    K_first = np.zeros(len(direction_pair[0]))
    K_second = np.zeros(len(direction_pair[1]))
        
    for dir_first, dir_second, i in zip(direction_pair[0], direction_pair[1], range(len(direction_pair[0]))):
        first_confusions = first_condition_data[condition_pair[0]]['Confusions'][dir_first]
        second_confusions = second_condition_data[condition_pair[1]]['Confusions'][dir_second]

        if condition_pair[0] == 'StaticOpenEars' or condition_pair[0] == 'DynamicOpenEars':
            first_confusions_mean = first_confusions[SUBJ_IDCS[0]]
        else:
            first_confusions_mean = np.nanmean(np.array([first_confusions[SUBJ_IDCS[0]], first_confusions[SUBJ_IDCS[0]+16]]), axis=0)

        if condition_pair[1] == 'StaticOpenEars' or condition_pair[1] == 'DynamicOpenEars':
            second_confusions_mean = second_confusions[SUBJ_IDCS[1]]
        else:
            second_confusions_mean = np.nanmean(np.array([second_confusions[SUBJ_IDCS[1]], second_confusions[SUBJ_IDCS[1]+16]]), axis=0)

        n_first = np.sum(~np.isnan(first_confusions_mean)) # Number of local data points
        k_first = np.nansum(first_confusions_mean) # Number of confusion in local data points

        n_second = np.sum(~np.isnan(second_confusions_mean)) # Number of local data points
        k_second = np.nansum(second_confusions_mean) # Number of confusion in local data points

        N_first[i] = n_first
        K_first[i] = k_first
        N_second[i] = n_second
        K_second[i] = k_second

    k = int(np.round(np.sum(K_second)))
    n = int(np.round(np.sum(N_second)))

    p_first = float(np.round(np.sum(K_first))) / np.sum(N_first)
    p_second = float(np.round(np.sum(K_second))) / np.sum(N_second)

    res = stats.binomtest(k, n, p_first, alternative='two-sided')
    print(#'Directions: ' + str(directions), 
          'Conditions: ' + str(condition_pair), 
          ' --> ' + str(int(round(p_first, 2) * 100)) + '%' + ' vs. ' +  str(int(round(p_second, 2) * 100)) +'%', 
          'pvalue: ' + str(res.pvalue), ' test-statistic: ' + str(k) + '/' + str(n))
    return


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
            res, T_plus, T_minus = stats_wilcoxon(confusion_rate_first, confusion_rate_second, alternative='two-sided', method='auto')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 'T(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(T_plus - T_minus), 'r = ' + str(effect_size))
        else:
            res = stats.ttest_rel(confusion_rate_first, confusion_rate_second, alternative='two-sided')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 't(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]))
    else:
        if NONPARAM:
            res, effect_size = stats_mannwhitneyu(confusion_rate_first, confusion_rate_second, alternative='two-sided')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 'U1(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]), 'r = ' + str(effect_size))
        else:
            res = stats.ttest_ind(confusion_rate_first, confusion_rate_second, alternative='two-sided')
            print(#'Directions: ' + str(directions), 
                'Conditions: ' + str(condition_pair), 
                ' --> ' + str(int(round(np.mean(confusion_rate_first), 2) * 100)) + '%' + ' vs. ' +  str(int(round(np.mean(confusion_rate_second), 2) * 100)) +'%', 
                'pvalue: ' + str(res[1]), 't(' + str(len(SUBJ_IDCS[0])) + ') = ' + str(res[0]))


    if 0:#condition_pair[0] == 'StaticKU100HRTF' and condition_pair[1] == 'DynamicKU100HRTF':
        plt.figure()
        plt.boxplot([confusion_rate_first, confusion_rate_second])
        plt.show(block=True)
    return res[1]

    

def plotLateralPlanes(idcs_list, pairtest_list, target_azi_list, name_list,
                       deg_list, title_bool_list, titles, xaxis_bool_list,
                       final_dict_names, local_azi_ele_data, coord_x, coord_y,
                       all_colors, EXP, root_dir, plot_avg_ele, ALL_PLANES):
    

    num_rows = 1
    num_cols = 4
    figsize = 3

    if ALL_PLANES:
        fig, axs = plt.subplots(nrows=num_rows,
                                ncols=num_cols,
                                sharey=True,
                                figsize=(4 * figsize, 1.1 * figsize),
                                gridspec_kw={
                                    'hspace': 0.1,
                                    'wspace': 0.05
                                })
    
    layer_idx = 0
    zorder_layers = [4,3,2]

    if ALL_PLANES:
        x_offs = np.array([1,0,-1]) * 7.5
    else:
        x_offs = np.array([0,0,0])

    all_confusion_rates = [[],[],[],[]]
    for mvp_idcs, pairs_to_be_tested, target_azimuths, name, deg, title_bool, xaxis_bool, target_distances in zip(
            idcs_list, pairtest_list, target_azi_list, name_list_azi, deg_list,
            title_bool_list, xaxis_bool_list, target_distance_hits_azi):
        
        azi_lim = 190
        azi_lim_pos_x = 230

        if not ALL_PLANES:
            fig, axs = plt.subplots(nrows=num_rows,
                                    ncols=num_cols,
                                    sharey=True,
                                    figsize=(4 * figsize, 1.1 * figsize),
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
            median_ratings = np.zeros(target_azimuths.size)
            azimuth_ratings = []

            if not ALL_PLANES:
                conditions_azi = np.asarray(
                    local_azi_ele_data[condition])[mvp_idcs, :, 0]
                if col >= 1:
                    cond_azi_first = np.asarray(
                        local_azi_ele_data[condition])[mvp_idcs, :, 0]
                    cond_azi_second = np.asarray(
                        local_azi_ele_data[condition])[mvp_idcs + 25, :, 0]
                    stacked = np.array([cond_azi_first, cond_azi_second])
                    conditions_azi = np.nanmean(stacked, axis=0)
                    
                conditions_azi[conditions_azi < -170.0] = (conditions_azi[conditions_azi < -170.0] + 360.0) % 360.0
                slope, intercept, sigma, bias = computeMetrics(
                conditions_azi, target_azimuths, median_statistic=True)
                axs[col].text(x=-30,
                            y=90,
                            s=r'$g = $' + str(round(slope, 2)),
                            fontsize=10)
                x = np.linspace(180, -180, 100)
                axs[col].plot(x,
                            slope * x + intercept,
                            zorder=0,
                            color='gray',
                            ls=':')

            markerstyles = ['v'] * 8 + ['D'] *8 +  ['^'] * 4
            confusion_rates = np.zeros(mvp_idcs.size)
            for i, j in zip(range(mvp_idcs.size), mvp_idcs):
                azimuth_ratings = local_azi_ele_data[condition][
                    mvp_idcs[i]][:, 0]
                if col >= 1:
                    azimuth_ratings = np.hstack((azimuth_ratings, local_azi_ele_data[condition][mvp_idcs[i] + 25][:, 0]))
                
                median_ratings[i] = np.nanmedian(azimuth_ratings)

                azimuth_scatter = np.copy(azimuth_ratings)
                azimuth_scatter[azimuth_ratings < -170.0] = (azimuth_scatter[azimuth_ratings < -170.0] + 360.0) % 360.0
                axs[col].scatter(
                    target_azimuths[i] +
                    (np.random.rand(azimuth_ratings.size) - 0.5) * 0 + x_offs[layer_idx],
                    #(np.random.rand(azimuth_ratings.size) - 0.5) * 2.5 + x_offs[layer_idx],
                    azimuth_scatter,
                    alpha=1.0,
                    edgecolors='k',
                    #marker=markerstyles[j],
                    s=15,
                    zorder=zorder_layers[layer_idx],
                    c=all_colors[mvp_idcs[i]])
                
                azimuth_ratings = azimuth_ratings[~np.isnan(azimuth_ratings)]
                if target_azimuths[i] == 180:
                    azimuth_ratings = (azimuth_ratings + 360.0) % 360.0 
                azimuth_distances = azimuth_ratings - target_azimuths[i]
                confusions = np.logical_or(azimuth_distances < target_distances[i,0],  azimuth_distances > target_distances[i,1])
                #confusions = np.logical_or(azimuth_distances < target_distances_fixed[0],  azimuth_distances > target_distances_fixed[1])
                confusion_rate = float(confusions.sum()) / float(confusions.size)
                confusion_rates[i] = confusion_rate

                # Confusion rates per direction
                if not ALL_PLANES:
                    axs[col].text(225, 220, s='LCR:', fontsize=9, style='italic')
                    axs[col].text(target_azimuths[i] - 2, 220, s=str(round(confusion_rate*100)), fontsize=9, style='italic')
            all_confusion_rates[col].append(confusion_rates)

            if not ALL_PLANES:
                axs[col].text(x=-30,
                        y=150,
                        s=r'$\overline{\mathrm{LCR}} = $' + str(round(np.mean(confusion_rates*100))) + '%',
                        fontsize=10)
                if col == 0:
                    axs[col].set_ylabel('Azimuth Response (deg.)')
                    renderInsetAxis(axs[col],
                            mvp_idcs,
                            coord_x,
                            coord_y,
                            [0.02, 0.585, 0.4, 0.4],
                            mkr_size=20)
            else:
                if col == 0 and layer_idx == 0: # plot only once all LS directions
                    axs[col].set_ylabel('Azimuth Response (deg.)')
                    if EXP == 'Static':
                        renderInsetAxis(axs[col],
                                np.arange(20),
                                coord_x,
                                coord_y,
                                #[-0.7, 0.3, 0.38, 0.38],
                                [0.004, 0.605, 0.38, 0.38],
                                #[-0.7, 0.585, 0.4, 0.4],
                                mkr_size=20)
                
            grid_azimuths = np.array([180, 150, 90, 30, 0, -30, -90, -150, -180])
            for i in range(grid_azimuths.size):
                # horizontal line
                axs[col].plot([-azi_lim_pos_x, azi_lim_pos_x], [grid_azimuths[i], grid_azimuths[i]], zorder=0, lw=0.5, color='gray', ls=(0, (1, 3)))
                # vertical line
                axs[col].plot([grid_azimuths[i], grid_azimuths[i]], [-azi_lim_pos_x, azi_lim_pos_x], zorder=0, lw=0.5, color='gray', ls=(0, (1, 3)))
            
            axs[col].set_xlim(-azi_lim, azi_lim_pos_x)
            axs[col].set_ylim(-170, azi_lim_pos_x)
            axs[col].invert_xaxis()
            axs[col].invert_yaxis()
            axs[col].plot([azi_lim, -azi_lim], [azi_lim, -azi_lim], color='grey', zorder=1)

            if xaxis_bool:
                #axs[col].set_xticks([-30, 0, 30])
                #[150, 90, 30, 0, -30, -90, -150]
                axs[col].set_xticks([150, 90, 30, 0, -30, -90, -150])
                axs[col].set_xlabel('Azimuth Target (deg.)')
            else:
                axs[col].set_xticks([150, 90, 30, 0, -30, -90, -150])
                axs[col].set_xticklabels([])

            axs[col].set_yticks([180, 150, 90, 30, 0, -30, -90, -150])
            if title_bool:
                axs[col].set_title(titles[col])
            #axs[col].grid(True)
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
        layer_idx += 1

        #plt.show(block=True)
        if not ALL_PLANES:
            plt.savefig(fname=pjoin(
                root_dir, 'Figures',
                name + '_LateralPlane_' + EXP.upper() + '.png'), bbox_inches='tight', dpi=300)

    if ALL_PLANES:
        # Plot global average confusion rates in case of combined plot
        GLOBAL_CR_STATIC = [4,13,8,11]
        GLOBAL_CR_DYNAMIC = [0,0,1,0]

        for col, condition in zip(range(num_cols), conditions):
            # confusion_rates_col = all_confusion_rates[col]

            # CR_L1 = confusion_rates_col[0]
            # CR_L2 = confusion_rates_col[1]
            # CR_L3 = confusion_rates_col[2]

            # CR_stacked = np.hstack((CR_L1,CR_L2,CR_L3)) 

            #CR_GLOBAL_MEAN = np.mean(CR_stacked)

            if EXP=='Static':
                CR_GLOBAL_MEAN = GLOBAL_CR_STATIC
            else:
                CR_GLOBAL_MEAN = GLOBAL_CR_DYNAMIC
            axs[col].text(x=-30,
                        y=150,
                        s=r'$\overline{\mathrm{LCR}} = $' + str(round(CR_GLOBAL_MEAN[col])) + '%',
                        fontsize=10)
            
        # Plot global accuracy measures in case of combined plot
        for col, condition in zip(range(num_cols), conditions):
            idcs = np.hstack((idcs_list_azi[0], idcs_list_azi[1], idcs_list_azi[2]))
            conditions_azi = np.asarray(
                local_azi_ele_data[condition])[idcs, :, 0]
            if col >= 1:
                cond_azi_first = np.asarray(
                    local_azi_ele_data[condition])[idcs, :, 0]
                cond_azi_second = np.asarray(
                    local_azi_ele_data[condition])[idcs + 25, :, 0]
                stacked = np.array([cond_azi_first, cond_azi_second])
                conditions_azi = np.nanmean(stacked, axis=0)
                
            conditions_azi[conditions_azi < -170.0] = (conditions_azi[conditions_azi < -170.0] + 360.0) % 360.0

            target_azimuths = np.hstack((target_azi_list[0], target_azi_list[1], target_azi_list[2])) 
            slope, intercept, sigma, bias = computeMetrics(
            conditions_azi, target_azimuths, median_statistic=True)
            axs[col].text(x=-30,
                        y=90,
                        s=r'$g = $' + str(round(slope, 2)),
                        fontsize=10)
            x = np.linspace(180, -180, 100)
            axs[col].plot(x,
                        slope * x + intercept,
                        zorder=0,
                        color='gray',
                        ls=':')
        
    #plt.show(block=True)
    if ALL_PLANES:
        plt.savefig(fname=pjoin(
            root_dir, 'Figures', 'LateralPlane_' + EXP.upper() + '.eps'), bbox_inches='tight', dpi=300)

def plotHemisphereMap(titles,
                      final_dict_names,
                      mean_azi_ele_data,
                      azi_ele_data,
                      plot_idcs,
                      QE_data,
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
    data_to_plot (string): 'Localization', 'QERate', or 'ResponseTime'.
    """

    # Plot localization plane data
    NUM_CHANNELS = 25
    #plot_idcs = np.arange(NUM_CHANNELS)
    channels = range(0, NUM_CHANNELS)
    num_rows = 1
    num_cols = 4
    ring_color = 'grey'
    figsize = 3
    offs = 0.04
    r = 0.8

    DRAW_CHANNEL_NUMBER = False
    NORM_RATE = 50  # QE Norm Constant

    if data_to_plot == 'ResponseTime':
        cmap = cm.get_cmap('afmhot_r')
    if data_to_plot == 'QERate':
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

    markeredge_colors = all_colors

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

        if data_to_plot == 'QERate':
            axs[col].scatter(r * coord_x,
                             r * coord_y,
                             facecolors=cmap(QE_data[condition] *
                                             100.0 / NORM_RATE),
                             edgecolors='k',
                             s=100,
                             alpha=1.0,
                             zorder=2)
        if data_to_plot == 'ResponseTime':
            if col == 0 and EXP == 'Static':
                idx += 1
                ext.append([
                    axs[col].get_window_extent().x0,
                    axs[col].get_window_extent().width
                ])
                axs[col].set_ylim(-1, 1)
                axs[col].set_xlim(-1, 1)
                axs[col].set_yticks([])
                axs[col].set_xticks([])
                continue
            else:
                val = (np.median(time_data[condition], axis=0) -
                       SEC_OFF) / NORM_SEC
                axs[col].scatter(r * coord_x,
                                 r * coord_y,
                                 facecolors=cmap(val),
                                 edgecolors='k',
                                 s=100,
                                 alpha=1.0,
                                 zorder=2)

        if data_to_plot == 'Localization':
            DRAW_RAW_RESPONSES = True
            if DRAW_RAW_RESPONSES:
                #rnd_cls = rand_cls[col]
                num_participants = azi_ele_data[condition].shape[0]
                for participant in range(num_participants):

                    if col > 0:
                        idcs = np.concatenate((plot_idcs, plot_idcs+25))
                        cls = np.concatenate((plot_colors, plot_colors))
                    else:
                        idcs = plot_idcs
                        cls = plot_colors
                        
                    azi = azi_ele_data[condition][participant, idcs, 0]
                    ele = azi_ele_data[condition][participant, idcs, 1]

                    loc_data_x, loc_data_y = convertAziEleToPlane(azi, ele)
                    axs[col].scatter(r * loc_data_x,
                                    r * loc_data_y,
                                    edgecolors='k',
                                    s=10,
                                    alpha=0.3,
                                    zorder=2,
                                    c=cls,
                                    marker='o')

            axs[col].scatter(r * np.delete(coord_x,plot_idcs),
                             r * np.delete(coord_y,plot_idcs),
                             facecolors='white',
                             edgecolors='k',
                             s=100,
                             alpha=0.6,
                             zorder=2)
            axs[col].scatter(r * coord_x[plot_idcs],
                             r * coord_y[plot_idcs],
                             facecolors=plot_colors,
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
            for n, i in zip(plot_idcs, range(plot_idcs.size)):
                axs[col].plot([r * coord_x[n], r * loc_data_x[i]],
                              [r * coord_y[n], r * loc_data_y[i]],
                              color='k',#mkr_cls[i],
                              alpha=0.8,
                              zorder=1,
                              ls=':')

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

    if data_to_plot == 'QERate':
        color_bar_ax = fig.add_axes([0.41, 0.05, 0.2, 0.05])
        norm = colors.Normalize(vmin=0, vmax=NORM_RATE, clip=True)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=color_bar_ax,
                            orientation="horizontal")
        cbar.set_label('Quadrant Errors (%)', rotation=0)
    if data_to_plot == 'ResponseTime':
        color_bar_ax = fig.add_axes([0.41, 0.05, 0.2, 0.05])
        norm = colors.Normalize(vmin=SEC_OFF, vmax=NORM_SEC+SEC_OFF, clip=True)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=color_bar_ax,
                            orientation="horizontal")
        cbar.set_label('Median Response Time (sec.)', rotation=0)
        cbar.set_ticks(np.arange(3,14))

    if data_to_plot == 'QERate':
        savename = 'Hemisphere_QuadrantErrors_' + EXP.upper() + '.pdf'
    if data_to_plot == 'Localization':
        savename = 'Hemisphere_Localization_' + EXP.upper() + '.png'
    if data_to_plot == 'ResponseTime':
        savename = 'Hemisphere_ResponseTime_' + EXP.upper() + '.pdf'

    plt.savefig(fname=pjoin(root_dir, 'Figures', savename),
                bbox_inches='tight', dpi=600)
    plt.show(block=True)
    return


def plotResponseTimesQuantitative(time_data, EXP, real_dict_names,
                                  final_dict_names, xticklabels, coord_x,
                                  coord_y, root_dir):

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
    dir_sets = [
        np.arange(25),
        np.arange(8),
        np.array([1, 2, 9, 10]),
        np.array([6, 7, 14, 15, 22, 21])
    ]

    fig, axs = plt.subplots(nrows=1,
                            ncols=len(dir_sets),
                            figsize=(len(dir_sets) * size, 1 * size),
                            sharey=False)
    x_offsets = [1, 2, 3, 4]

    for dirs, col in zip(dir_sets, range(len(dir_sets))):

        POOL_DIR_DATA = False
        #SEC_OFF = 3.0

        if POOL_DIR_DATA:
            data = np.zeros((len(conditions), 16 * dirs.size))
            for condition, idx in zip(conditions, range(len(conditions))):
                data[idx, :] = (time_data[condition][:, dirs]).flatten()
        else:
            data = np.zeros((len(conditions), 16))
            for condition, idx in zip(conditions, range(len(conditions))):
                data[idx, :] = np.median(time_data[condition][:, dirs], axis=1)
        pairs_to_be_tested = [[1, 2], [1, 3]]
        pvals, effect_sizes, T, z = posthoc_wilcoxon(data,
                                               pairs_to_be_tested,
                                               alternative_h='two-sided',
                                               p_adjust='Bonferroni',
                                               CL_ES='lesser')
        for dict_name, idx, x_off in zip(conditions, range(len(conditions)),
                                         x_offsets):
            if EXP == 'Static' and dict_name == 'StaticOpenEars':
                continue
            dir_mean_resp_times = data[idx, :]
            median_time = np.median(dir_mean_resp_times)  # participant median
            median_int = median_conf_int()
            median_int.low = np.quantile(dir_mean_resp_times, q=0.25)
            median_int.high = np.quantile(dir_mean_resp_times, q=0.75)
            asymmetric_IQR = np.array(
                [median_time - median_int.low,
                 median_int.high - median_time])[:, None]
            axs[col].errorbar(x_off,
                              median_time,
                              capsize=2.0,
                              linestyle='none',
                              xerr=0,
                              yerr=asymmetric_IQR,
                              color='k',
                              zorder=2)
            axs[col].plot(x_off,
                          median_time,
                          's',
                          markersize=6,
                          markerfacecolor='white',
                          markeredgecolor='k',
                          zorder=3)
            #axs[col].scatter(x_off + (np.random.rand(dir_mean_resp_times.size) - 0.5) * 0.2, dir_mean_resp_times, s=5, c='white', edgecolors='gray', zorder=1)
        axs[col].text(1.15, -0.7 + 3.0, s='Real')
        axs[col].text(3, -0.7 + 3.0, s='Virtual')

        y_refs = [2 + 3.0, 1.5 + 3.0]
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

                axs[col].plot([brace_start, brace_end], [y_ref, y_ref],
                              color='k')
                axs[col].plot([brace_start, brace_start], [y_ref, y_ref + 0.1],
                              color='k')
                axs[col].plot([brace_end, brace_end], [y_ref, y_ref + 0.1],
                              color='k')
                axs[col].text((brace_start + brace_end) / 2 + xoffs,
                              y_ref + yoffs,
                              s=pval_str,
                              fontsize=10)

                # if np.abs(effect_sizes[i]) >= 0.75:
                #     eff_str = r'$\Delta$'
                #     xoffs = -0.075
                #     yoffs = 0.1
                #     axs[col].text((brace_start + brace_end) / 2 + xoffs,
                #                   y_ref + yoffs,
                #                   s=eff_str,
                #                   fontsize=10)

        axs[col].set_xlim(0, 5)
        if col == 0:
            axs[col].set_ylabel('Response time (sec.)')
        # plt.xlabel('Condition')
        axs[col].set_xticks(x_offsets, xticklabels)
        # axs[col].set_yticks(np.arange(0, 11), ['0', '', '2',
        #                    '', '4', '', '6', '', '8', '', '10'])
        #axs[col].set_yticks(np.arange(1, 12),
        #                    ['1', '', '3', '', '5', '', '7', '', '9', '', ''])
        axs[col].set_yticks(np.arange(4, 14))
        # axs[col].set_ylim(1, 11)
        axs[col].set_ylim(4, 14)
        axs[col].grid(axis='y')

        if EXP == 'Dynamic':
            pos_size_list = [0.015, 0.61, 0.4, 0.4]
        else:
            pos_size_list = [0.015, 0.1, 0.4, 0.4]

        renderInsetAxis(axs[col],
                        dirs,
                        coord_x,
                        coord_y,
                        pos_size_list,
                        mkr_size=15)
    # plt.show(block=True)
    plt.savefig(fname=pjoin(root_dir, 'Figures',
                            EXP + '_ResponseTimes_Quantitative.pdf'),
                bbox_inches='tight')
    print('')


def plotResponseTimeProgression(time_data_sorted, EXP):
    if EXP == 'Static':
        title = 'Response times over 150 static trials (Op.Headph., Indiv, KU100)'
    else:
        title = 'Response times over 150 dynamic trials (Op.Headph., KEMAR, KU100)'

    scale = 3
    plt.figure(figsize=(3*scale,1*scale))
    plt.fill_between(np.arange(150), np.median(time_data_sorted, axis=0) - MAD(time_data_sorted, axis=0), np.median(time_data_sorted, axis=0)+ MAD(time_data_sorted, axis=0), color='lightgrey')
    plt.errorbar(x=np.arange(150), y=np.median(time_data_sorted, axis=0), yerr=0, label='median +- MAD')
    plt.title(title)
    plt.ylabel('Response Time (sec.)')
    plt.xlabel('Trial Index')
    plt.xlim(0,150)
    plt.xticks(np.arange(0,165,15))
    plt.legend()
    plt.ylim(0, 20)
    plt.grid()
    plt.tight_layout()

    if EXP == 'Static':
        savename = 'ResponseTimes_Progression_Static.png'
    else:
        savename = 'ResponseTimes_Progression_Dynamic.png'

    plt.savefig(pjoin('Figures', savename))
    plt.show()

