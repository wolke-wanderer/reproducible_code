# Define the __all__ variable for "from ... import *"
__all__ = ["jokes","configure"]
# TO-DO: check if this needs another import logging for * / do not use import *

# Import the submodules
from . import jokes
from . import configure
# Import the external logging module
import logging
