"""Custom mjlab tasks. Imported by mjlab via the ``mjlab.tasks`` entry point."""

from pathlib import Path

# Asset modules use this path to locate files shipped inside the package.
ESRA_MJLAB_SRC_PATH: Path = Path(__file__).parent

# Importing the robot configuration registers the task with mjlab.
from esra_unified_mjlab_simulator.cfg import leo_quadruped as leo_quadruped  # noqa: E402
