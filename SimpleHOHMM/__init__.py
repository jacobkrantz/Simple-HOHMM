import json
from os.path import dirname

from .builder import HiddenMarkovModelBuilder
from .model import HiddenMarkovModel

with open(dirname(__file__) + '/package_info.json') as f:
    _info = json.load(f)

__version__ = str(_info["version"])
__author__ = str(_info["author"])
__contact__ = str(_info["author_email"])
