REGISTRY = {}
from .dot_role import DotRole
REGISTRY["dot"] = DotRole

from .q_role import QRole
REGISTRY["q"] = QRole