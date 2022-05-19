import numpy as np
import os
import logging
from functools import reduce
import yaml

def compute_seq_id(hit, or_id=0):
    """Compute the RICH PMT sequence ID"""
    disk_id, pm_id, sc_id, up_dw_id, _ = hit
    if or_id < 1:
        seq_id = sc_id * 8 + pm_id + up_dw_id * 61 * 8 + disk_id * 61 * 8 * 2
    else:
        seq_id = 61 * 8 * 2 * 2 + sc_id + up_dw_id * 61 + disk_id * 61 * 2
    return int(seq_id)

compute_seq_id = np.vectorize(compute_seq_id, otypes=[int])

def get_config(key=None, config_file="configs/config.yaml"):
    """
    Read the configuration file and return value of the key if present

    Args:
        key (str): Access specified key values (Format: "foo.bar.z")

    Returns:
        conf: Value for the specified key else dictionary of config_file contents
    """
    global yaml
    with open(config_file, "r") as conf:
        try:
            conf = yaml.safe_load(conf)
        except yaml.YAMLError as err:
            print("Error reading config file: {}".format(err))
    if key:
        conf = reduce(lambda c, k: c[k], key.split("."), conf)
    return conf

def get_logger(file_path=None, file_name=None):
    """
    This function initialize the log file

    Parameters
    ----------
    file_path: path were the logs are stored
    file_name: name of the log file

    Returns
    -------
    log: log configuration
    """

    logging.getLogger("py4j").setLevel(logging.ERROR)
    log = logging.getLogger("main_logger")
    log.setLevel("INFO")

    if not log.handlers:

        formatter = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s")
        
        # Create file handler
        if file_path:
            logger_filepath = os.path.join(file_path, file_name)
            os.makedirs(file_path, exist_ok=True)  # create folder if needed
            fh = logging.FileHandler(logger_filepath)
            fh.setLevel("INFO")
            fh.setFormatter(formatter)
            log.addHandler(fh)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel("INFO")
        ch.setFormatter(formatter)
        log.addHandler(ch)

    return log