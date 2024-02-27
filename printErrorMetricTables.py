from Utility.localizationEvaluationUtility import loadAndPrintErrorMetric, dirset_names
from os.path import dirname, join as pjoin
import numpy as np
root_dir = dirname(__file__)

PRINT_ANGULAR_ERRORS = False
if PRINT_ANGULAR_ERRORS:
    # Requires first saving metric data by executing the evaluation scripts
    # and setting the flag SAVE_ERROR_METRICS = True
    loadAndPrintErrorMetric(pjoin(root_dir, 'ErrorMetricData'), dirset_names)

# Quadrant-Error Table Data for paper revision
EXPS = ['Static', 'Dynamic']

for dirset_name in ['Frontal', 'Rear', 'Overall']:
    for EXP in EXPS:
        metric_data_all = np.load(file=pjoin(
                            pjoin(root_dir, 'ErrorMetricData'),
                            EXP + '_' + dirset_name + '_' + 'QuadrantError' + '.npy'),
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
            print(str(round(metric_data_all[dict_name] * 100.0, 2)) + ' & ', end="")
            if dict_name == 'DynamicKU100HRTF':
                print(' \\\\')

    print('')