# This conftest.py is for the ST test framework
# It loads the fixtures from fixtures.py

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.st.framework.fixtures import *  # noqa: F401, F403
