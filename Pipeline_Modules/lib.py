"""
Library file
Import all built-in modules and third-party libraries only once
"""
#  built-in modules
import os
import time
import argparse
import subprocess
import sys
import traceback
import socket
import textwrap
import glob
import re
import csv
import shutil
import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path


import pyvista as pv

# libraries
import pyvirtualdisplay
env_vars = dict(os.environ)
if "DISPLAY" in env_vars:
    pass
else:
    display = pyvirtualdisplay.Display(visible=0, size=(1280, 1024))
    display.start()

import mayavi.mlab as mlab
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import SimpleITK as sitk
import cc3d
import scipy.signal as signal
import scipy.ndimage.morphology as morphology
import scipy.spatial.transform as scipy_trans
import scipy.spatial as spatial
import pandas as pd

import sklearn.decomposition as decomposition
import sklearn.preprocessing
import skimage.measure as measure

from fpdf import FPDF
from PyPDF2 import PdfFileMerger, PdfFileReader
from zipfile import ZipFile
import yaml
import skimage

