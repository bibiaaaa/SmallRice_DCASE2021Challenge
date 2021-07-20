import yaml
from tqdm import tqdm
import numpy as np
import json
import csv
from pprint import pformat
import sys
import pathlib
from torchaudio.transforms import MelScale, MelSpectrogram
from loguru import logger
import math
from typing import Callable, Optional

import torch
from torch import Tensor
from torchaudio import functional as F
from torchaudio.compliance import kaldi


def save_dict(d, path):
    with open(path, 'w') as f:
        f.write(pretty_dict(d))


def save_csv(c, path):
    with open(path, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(c)


def pretty_dict(d):
    return json.dumps(d, indent=4, ensure_ascii=False)


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file, 'r') as con_read:
        config_parameters = yaml.load(con_read, Loader=yaml.FullLoader)
    arguments = dict(config_parameters, **kwargs)
    for key, value in DEFAULT_ARGS.items():
        arguments.setdefault(key, value)
    return arguments


def getfile_outlogger(outputfile):
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])
    if outputfile:
        logger.add(outputfile, enqueue=True, format=log_format)
    return logger


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)
