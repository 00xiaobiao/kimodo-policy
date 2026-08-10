"""Load the HumanoidArena Conda packages into Isaac Sim's bundled Python."""

import os
import site


for variable in ("KIMODO_SIM_SITE_PACKAGES", "KIMODO_HUMANOIDARENA_ISAACLAB"):
    path = os.environ.get(variable, "").strip()
    if path:
        site.addsitedir(path)
