version = "0.0.1"
date = "2015-06"
import sys, logging
logging.basicConfig()

if sys.version_info < (2, 6):
    logger = logging.getLogger("xrfit.__init__")
    logger.error("xrfit requires a python version >= 2.6")
    raise RuntimeError("xrfit requires a python version >= 2.6, now we are running: %s" % sys.version)

#from peak_fit import PeakFit
from multi_peak_fit import MultiPeakFit
