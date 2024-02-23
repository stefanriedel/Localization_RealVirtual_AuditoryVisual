import json
import numpy as np
from squaternion import Quaternion
from os.path import dirname, join as pjoin
import os
from Utility.ambisonics import sph2cart, cart2sph
from Utility.piercingpoint import find_piercing_point, convert_piercing_to_azi_ele, plotTriangulation
from Utility.loudspeakerPositions import azi, ele, azi_hull, ele_hull, xyz, xyz_hull
import scipy.spatial as spatial
from Utility.localizationEvaluationUtility import *

from datetime import datetime

USE_PIERCINGPOINT_DIRECTION = True

RENDER_LATERAL_PLANES = False
ALL_PLANES = True # Plot all planes in one plot instead of separate plots

RENDER_VERTICAL_PLANES = True

RENDER_HEMI_MAP = False
RENDER_TIME_DATA_PLOT = False

GEOMETRIC_MEDIAN_RESPONSE = True
SAVE_ERROR_METRICS = True

NUM_CHANNELS = 25

RENDER_WITH_JASA_NAMES = True

# Tolerance around target direction to consider as hit
# Otherwise it is a 'quadrant error'
ANGLE_TOL = 90.0 / 180.0 * np.pi

# Opening JSON file
root_dir = dirname(__file__)
data_dir = pjoin(root_dir, 'Responses/results_dynamic')
figures_dir = pjoin(root_dir, 'Figures', 'ExperimentResults')

file_list = os.listdir(data_dir)

if '.DS_Store' in file_list:
    file_list.remove('.DS_Store')
if 'hidden_results' in file_list:
    file_list.remove('hidden_results')

num_participants = len(file_list)
results_dynamic_open_ears = []
results_dynamic_headphones = []

ages = []
genders = []
exp_times_min = []

presented_order_trials_hp = []

for subj in range(num_participants):
    f_location = pjoin(data_dir, file_list[subj])

    f = open(f_location)
    # returns JSON object as
    # a dictionary
    json_file = json.load(f)

    ages.append(json_file['SubjectInformation']['Age'])
    genders.append(json_file['SubjectInformation']['Gender'])

    start_time = datetime.strptime(json_file['StartTime'], "%d %b %Y %H:%M:%S")
    end_time = datetime.strptime(json_file['EndTime'], "%d %b %Y %H:%M:%S")

    exp_time_seconds = (end_time - start_time)
    exp_times_min.append(int(exp_time_seconds.total_seconds() / 60.0))

    data = json_file['Results']['Parts']

    # Data of the three parts
    results_dynamic_open_ears.append(data[0]['Trials'])
    results_dynamic_headphones.append(data[1]['Trials'])

    presented_order_trials_hp.append(data[1]['PresentedOrderOfTrials'])

mean_age = np.mean(np.asarray(ages))
stddev_age = np.std(np.asarray(ages))
mean_time = np.mean(np.asarray(exp_times_min))


coord = np.array([np.asarray(azi, dtype=float),
                  np.asarray(ele, dtype=float)]).transpose()
target_coord_deg = np.copy(coord)

target_unit_vectors = sph2cart(coord[:, 0] / 180.0 * np.pi,
                               (90.0 - coord[:, 1]) / 180.0 * np.pi).T

coord_x, coord_y = convertAziEleToPlane(azi, ele)
channels = range(0, NUM_CHANNELS)

if USE_PIERCINGPOINT_DIRECTION:
    hull_unit = sph2cart(
        np.asarray(azi_hull) / 180.0 * np.pi,
        (90.0 - np.asarray(ele_hull)) / 180.0 * np.pi).T
    hull = spatial.ConvexHull(hull_unit)
    if 0:
        plotTriangulation(hull, xyz_hull)

dict_names = ['DynamicOpenEars', 'DynamicHeadphones']
headphone_dict_names = [
    'DynamicOpenHeadphones', 'DynamicKU100HRTF', 'DynamicKEMARHRTF'
]
part_results = [results_dynamic_open_ears, results_dynamic_headphones]
part_num_trials = [25, 150]

# Create empty dictionaries
azi_ele_data_parts = {dict_name: np.array([]) for dict_name in dict_names}
time_data_parts = {dict_name: np.array([]) for dict_name in dict_names}

for (part_result, dict_name, num_trials) in zip(part_results, dict_names,
                                                part_num_trials):
    azi_ele_data_part = np.zeros((num_participants, num_trials, 2))
    time_data_part = np.zeros((num_participants, num_trials))

    # Get results of part one
    for participant in range(num_participants):
        for tr in range(num_trials):
            all_data = part_result[participant][tr]['Ratings']
            # raw_ypr = all_data[1:4]
            raw_quat = all_data[-4:]
            converted_quat = [
                raw_quat[3], raw_quat[2], raw_quat[0], raw_quat[1]
            ]
            # quaternion class requires w,x,y,z
            q = Quaternion(converted_quat[0], converted_quat[1],
                           converted_quat[2], converted_quat[3])
            # convert to roll, pitch, yaw
            e = q.to_euler(degrees=True)
            roll = e[0]
            elevation = e[1] * -1
            azimuth = e[2]

            if USE_PIERCINGPOINT_DIRECTION:
                # pointing direction of gun
                line_dir = sph2cart(azimuth / 180.0 * np.pi,
                                    (90.0 - elevation) / 180.0 * np.pi)
                # position of gun
                line_base = np.asarray(all_data[4:7])
                point, params = find_piercing_point(line_base, line_dir, hull,
                                                    xyz_hull)
                center = np.array([0, 0, 1.2])
                azimuth, elevation = convert_piercing_to_azi_ele(point,
                                                                 center,
                                                                 degrees=True)

            azi_ele_data_part[participant, tr, 0] = azimuth
            azi_ele_data_part[participant, tr, 1] = elevation

            elapsed_time = part_result[participant][tr]['ElapsedTimeInSeconds']
            time_data_part[participant, tr] = elapsed_time
    azi_ele_data_parts[dict_name] = azi_ele_data_part
    time_data_parts[dict_name] = time_data_part

final_dict_names = [
    'DynamicOpenEars', 'DynamicOpenHeadphones', 'DynamicKU100HRTF',
    'DynamicKEMARHRTF'
]
final_num_trials = [25, 50, 50, 50]

azi_ele_data = {dict_name: np.array([]) for dict_name in final_dict_names}
azi_ele_data['DynamicOpenEars'] = azi_ele_data_parts['DynamicOpenEars']
idcs = np.arange(NUM_CHANNELS)
azi_ele_data['DynamicOpenHeadphones'] = azi_ele_data_parts[
    'DynamicHeadphones'][:, idcs.tolist() + (idcs + 75).tolist(), :]
azi_ele_data['DynamicKU100HRTF'] = azi_ele_data_parts['DynamicHeadphones'][:, (
    idcs + 25).tolist() + (idcs + 75 + 25).tolist(), :]
azi_ele_data['DynamicKEMARHRTF'] = azi_ele_data_parts['DynamicHeadphones'][:, (
    idcs + 50).tolist() + (idcs + 75 + 50).tolist(), :]


# Evaluate ResponsetTime over trials

times_raw = time_data_parts['DynamicHeadphones']

times_raw_sorted = np.zeros((16,150))
for subj in range(16):
    times_raw_sorted[subj,:] = times_raw[subj, presented_order_trials_hp[subj]]

def MAD(data, axis=0):
    return np.nanmedian(np.abs(data - np.nanmedian(data, axis=axis)), axis=axis)


scale = 3
plt.figure(figsize=(2*scale,1*scale))
#plt.errorbar(x=np.arange(150), y=np.mean(times_raw_sorted, axis=0), yerr=np.std(times_raw_sorted, axis=0), label='mean +- STD')

plt.fill_between(np.arange(150), np.median(times_raw_sorted, axis=0) - MAD(times_raw_sorted, axis=0), np.median(times_raw_sorted, axis=0)+ MAD(times_raw_sorted, axis=0), color='lightgrey')
plt.errorbar(x=np.arange(150), y=np.median(times_raw_sorted, axis=0), yerr=0, label='median +- MAD')


#plt.title('Response Times over 150 Trials interleaving OpenHeadphones (Real), KEMAR (Virtual), and KU100 (Virtual)')
plt.title('Response times over 150 dynamic trials (Op.Headph., KEMAR, KU100)')
plt.ylabel('Response Time (sec.)')
plt.xlabel('Trial Index')
plt.xlim(0,150)
plt.xticks(np.arange(0,165,15))
plt.legend()
plt.ylim(0, 20)
plt.grid()
plt.tight_layout()

plt.savefig('Figures/ResponseTimes_Over_Time.eps')
plt.show()


time_data = {dict_name: np.array([]) for dict_name in final_dict_names}
time_data['DynamicOpenEars'] = time_data_parts['DynamicOpenEars']
idcs = np.arange(NUM_CHANNELS)
time_data['DynamicOpenHeadphones'] = time_data_parts[
    'DynamicHeadphones'][:, idcs.tolist() + (idcs + 75).tolist()]
time_data['DynamicKU100HRTF'] = time_data_parts['DynamicHeadphones'][:, (
    idcs + 25).tolist() + (idcs + 75 + 25).tolist()]
time_data['DynamicKEMARHRTF'] = time_data_parts['DynamicHeadphones'][:, (
    idcs + 50).tolist() + (idcs + 75 + 50).tolist()]

# Convert azi ele to unit vector data
unit_vector_data = {dict_name: np.array([]) for dict_name in final_dict_names}
for dict_name, num_trials in zip(final_dict_names, final_num_trials):
    unit_vectors = np.zeros((num_participants, num_trials, 3))
    for participant in range(num_participants):
        azi_rad = azi_ele_data[dict_name][participant, :, 0] / 180.0 * np.pi
        zen_rad = (90.0 -
                   azi_ele_data[dict_name][participant, :, 1]) / 180.0 * np.pi
        unit_vectors[participant, :, :] = sph2cart(azi_rad, zen_rad).T
    unit_vector_data[dict_name] = unit_vectors

# Compute confusion indices and rate, and mean directions of remaining points
confusion_data = {dict_name: np.array([]) for dict_name in final_dict_names}
mean_unit_vector_data = {
    dict_name: np.array([])
    for dict_name in final_dict_names
}
local_azi_ele_data = {
    dict_name: np.array([])
    for dict_name in final_dict_names
}

stacked_target_vectors = np.vstack((target_unit_vectors, target_unit_vectors))
target_unit_vector_list = [
    target_unit_vectors, stacked_target_vectors, stacked_target_vectors,
    stacked_target_vectors
]
for dict_name, num_trials, targ_unit_vecs in zip(final_dict_names,
                                                 final_num_trials,
                                                 target_unit_vector_list):
    confusion_percentage = np.zeros(num_trials)
    mean_unit_vectors = np.zeros((num_trials, 3))
    local_azi_ele = [np.array([])] * num_trials

    response_vectors = unit_vector_data[dict_name]
    for tr in range(num_trials):
        local_response_vectors = np.array([[0, 0, 0]])
        idx = 0.0
        confusion_counter = 0.0
        target_vec = targ_unit_vecs[tr, :]
        not_nan_idcs = []
        for participant in range(num_participants):
            response_vec = response_vectors[participant, tr, :]
            if (np.arccos(np.dot(target_vec, response_vec)) > ANGLE_TOL):
                confusion_counter += 1.0
                nan_vec = np.array([np.nan, np.nan, np.nan])
                local_response_vectors = np.concatenate(
                    (local_response_vectors, nan_vec[None, :]), axis=0)
            else:
                local_response_vectors = np.concatenate(
                    (local_response_vectors, response_vec[None, :]), axis=0)
                not_nan_idcs.append(participant)
            idx += 1.0
        confusion_percentage[tr] = confusion_counter / idx
        local_response_vectors = local_response_vectors[1:, :]
        azi, zen = cart2sph(local_response_vectors[:, 0],
                            local_response_vectors[:, 1],
                            local_response_vectors[:, 2])
        local_azi_ele[tr] = np.array(
            [azi * 180.0 / np.pi, (np.pi / 2 - zen) * 180.0 / np.pi]).T

        if GEOMETRIC_MEDIAN_RESPONSE:
            local_response_no_nans = local_response_vectors[not_nan_idcs, :]
            mean_unit_vec = geometric_median(local_response_no_nans)
        else:
            mean_unit_vec = np.nanmedian(local_response_vectors, axis=0)
        mean_unit_vec /= np.linalg.norm(mean_unit_vec)
        mean_unit_vectors[tr, :] = mean_unit_vec
    confusion_data[dict_name] = confusion_percentage
    mean_unit_vector_data[dict_name] = mean_unit_vectors
    local_azi_ele_data[dict_name] = local_azi_ele

targets_azi_ele = np.copy(target_coord_deg)



mean_azi_ele_data = {dict_name: np.array([]) for dict_name in final_dict_names}
for dict_name in headphone_dict_names:
    stacked_confusion = np.array(
        [confusion_data[dict_name][:25], confusion_data[dict_name][25:]])
    confusion_data[dict_name] = np.mean(stacked_confusion, axis=0)

    stacked_mean_vecs = np.array([
        mean_unit_vector_data[dict_name][:25, :],
        mean_unit_vector_data[dict_name][25:, :]
    ])
    mean_unit_vector_data[dict_name] = np.mean(stacked_mean_vecs, axis=0)

    stacked_time = np.array(
        [time_data[dict_name][:, :25], time_data[dict_name][:, 25:]])
    time_data[dict_name] = np.mean(stacked_time, axis=0)

if SAVE_ERROR_METRICS:
    computeAndSaveErrorMetrics(pjoin(
        root_dir,
        'ErrorMetricData',
    ), 'Dynamic', dir_sets, dirset_names, final_dict_names, local_azi_ele_data,
        targets_azi_ele, confusion_data, ABS_BIAS=False)

if RENDER_TIME_DATA_PLOT:
    EXP = 'Dynamic'
    real_dict_names = ['DynamicOpenEars', 'DynamicOpenHeadphones']
    xticklabels = ['REF', 'OH',
                   'KE', 'KU']
    plotResponseTimesQuantitative(
        time_data, EXP, real_dict_names, final_dict_names, xticklabels, coord_x, coord_y, root_dir)
    

# Lateral plane plots
if RENDER_LATERAL_PLANES:
    titles = ['Reference', 'Open Headphones', 'KEMAR HRIR', 'KU100 HRIR']
    EXP = 'Dynamic'
    plot_avg_ele = False
    plotLateralPlanes(idcs_list_azi, pairtest_list_azi, target_azi_list, name_list,
                       deg_list_azi, title_bool_list_azi, titles, xaxis_bool_list_azi, final_dict_names,
                       local_azi_ele_data, coord_x, coord_y, all_colors, EXP,
                       root_dir, plot_avg_ele, ALL_PLANES)

# Vertical plane plots
if RENDER_VERTICAL_PLANES:
    titles = ['Reference', 'Open Headphones', 'KEMAR HRIR', 'KU100 HRIR']
    EXP = 'Dynamic'
    plot_avg_ele = False
    if RENDER_WITH_JASA_NAMES:
        name_list_vertical = name_list_jasa_dynamic
    plotVerticalPlanes(idcs_list, pairtest_list, target_ele_list, name_list_vertical,
                       deg_list, title_bool_list, titles, xaxis_bool_list, final_dict_names,
                       local_azi_ele_data, coord_x, coord_y, all_colors, EXP,
                       root_dir, plot_avg_ele, RENDER_WITH_JASA_NAMES)

# Reassure normalized mean response vectors and convert to azi ele for hemi maps
for dict_name in final_dict_names:
    x = np.copy(mean_unit_vector_data[dict_name][:, 0])
    y = np.copy(mean_unit_vector_data[dict_name][:, 1])
    z = np.copy(mean_unit_vector_data[dict_name][:, 2])
    norm = np.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    azi, zen = cart2sph(x, y, z)
    mean_azi_ele_data[dict_name] = np.array(
        [azi * 180.0 / np.pi, (np.pi / 2 - zen) * 180.0 / np.pi]).T
    mean_unit_vector_data[dict_name][:, 0] = x
    mean_unit_vector_data[dict_name][:, 1] = y
    mean_unit_vector_data[dict_name][:, 2] = z
if RENDER_HEMI_MAP:
    titles = ['Reference', 'Open Headphones', 'KEMAR HRIR', 'KU100 HRIR']
    EXP = 'Dynamic'
    plots = ['ResponseTime'] #['Localization', 'ConfusionRate', 'ResponseTime']
    main_titles = [True, False, True]
    sub_titles = [True, False, True]
    for plot, main_title, sub_title, in zip(plots, main_titles, sub_titles):
        plotHemisphereMap(titles,
                          final_dict_names,
                          mean_azi_ele_data,
                          confusion_data,
                          time_data,
                          coord_x,
                          coord_y,
                          all_colors,
                          EXP,
                          root_dir,
                          main_title,
                          sub_title,
                          data_to_plot=plot)

print('done')
