import json
from pathlib import Path

# Print fps function, to check the network client frequency.
t = 0
t0 = 0
t1 = 0

CONFIG_DIR = Path(__file__).resolve().parents[5] / "configs"


def _load_norm_states():
    """
    Load normalization constants from configs/norm_states.json.
    Values were removed from the public release; users must supply their own.
    """
    config_path = CONFIG_DIR / "norm_states.json"
    if not config_path.exists():
        raise RuntimeError(
            "Core normalization constants were removed for the public release. "
            "Add configs/norm_states.json with the required numeric fields."
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


# Aircraft normalization values
NormStates = _load_norm_states()
