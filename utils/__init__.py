import os
from .log_manage import logger_init
from .data_helper import LoadEnglishGermanDataset

PROJECT_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_HOME = os.path.join(PROJECT_HOME, 'data')

__all__ = [
    'PROJECT_HOME',
    'DATA_HOME',
    'logger_init',
    'LoadEnglishGermanDataset'
]
