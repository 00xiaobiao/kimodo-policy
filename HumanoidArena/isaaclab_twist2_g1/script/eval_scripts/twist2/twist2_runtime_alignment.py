from __future__ import annotations

import os


TWIST2_TRAINING_PHYSICS_DT = 0.002
TWIST2_TRAINING_DECIMATION = 10


def align_twist2_env_cfg(env_cfg) -> tuple[float, int]:
    """Match the Isaac evaluation integration to TWIST2's training timestep.

    TWIST2's student controller runs at 50 Hz, with ten 2 ms physics steps per
    policy action.  A 5 ms/4-step setup has the same nominal control rate but
    different contact and PD-drive dynamics, which causes reference replay to
    drift and eventually fall.
    """

    physics_dt = float(os.getenv("TWIST2_PHYSICS_DT", TWIST2_TRAINING_PHYSICS_DT))
    decimation = int(os.getenv("TWIST2_DECIMATION", TWIST2_TRAINING_DECIMATION))
    if physics_dt <= 0.0:
        raise ValueError(f"TWIST2_PHYSICS_DT must be positive, got {physics_dt}")
    if decimation <= 0:
        raise ValueError(f"TWIST2_DECIMATION must be positive, got {decimation}")

    control_dt = physics_dt * decimation
    if abs(control_dt - 0.02) > 1e-8:
        raise ValueError(
            "TWIST2 evaluation must remain at 50 Hz: "
            f"physics_dt={physics_dt} decimation={decimation} gives control_dt={control_dt}"
        )

    env_cfg.sim.dt = physics_dt
    env_cfg.decimation = decimation
    env_cfg.sim.render_interval = decimation

    contact_forces = getattr(getattr(env_cfg, "scene", None), "contact_forces", None)
    if contact_forces is not None:
        contact_forces.update_period = physics_dt

    print(
        "[TWIST2 alignment] "
        f"physics_dt={physics_dt:g}s decimation={decimation} control_hz={1.0 / control_dt:g}"
    )
    return physics_dt, decimation
