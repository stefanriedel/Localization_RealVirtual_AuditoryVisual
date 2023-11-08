from Utility.localizationEvaluationUtility import loadAndPrintErrorMetric, dirset_names
from os.path import dirname, join as pjoin
root_dir = dirname(__file__)

# Requires first saving metric data by executing the evaluation scripts
loadAndPrintErrorMetric(pjoin(root_dir, 'ErrorMetricData'), dirset_names)
