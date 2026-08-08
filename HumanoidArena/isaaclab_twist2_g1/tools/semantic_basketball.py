"""Runtime helper for football semantic distractor basketballs."""

import os
from pathlib import Path


BASKETBALL_LOCAL_OFFSET = (2.463015057933452, -2.503791749264172, 0.2791186808994101)
DEFAULT_GOAL_ORIGIN = (-2.5, 3.0)
DEFAULT_GOAL_Z = 0.65
DEFAULT_BASKETBALL_MASS = 1.0


def ensure_semantic_basketballs(
    goal_origin: tuple[float, float] = DEFAULT_GOAL_ORIGIN,
    goal_z: float = DEFAULT_GOAL_Z,
    asset_path: str | None = None,
    force: bool = False,
) -> bool:
    """Spawn semantic basketballs as independent dynamic bodies.

    The old semantic goal USD carries the basketball as a child of GoalNet. GoalNet is
    intentionally spawned as a kinematic scene asset, so the child basketball can be
    treated as part of the static goal hierarchy. This helper disables that child and
    creates an independent per-env rigid body at the same initial world-local pose.
    """
    if not force and os.environ.get("TEST_MODE") != "semantic":
        return False

    try:
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        print(f"[semantic_basketball] skipped: cannot import USD modules: {exc}")
        return False

    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        project_root = str(Path(__file__).resolve().parents[1])
    if asset_path is None:
        asset_path = str(
            Path(project_root)
            / "assets"
            / "objects"
            / "small_warehouse"
            / "football_scene"
            / "interaction_obj"
            / "basketball"
            / "model_Sports_Basketball_B08QJJPWKT_Brown_69323.usd"
        )

    if not Path(asset_path).exists():
        print(f"[semantic_basketball] skipped: asset not found: {asset_path}")
        return False

    stage = omni.usd.get_context().get_stage()
    envs_prim = stage.GetPrimAtPath("/World/envs")
    if envs_prim and envs_prim.IsValid():
        env_paths = [str(child.GetPath()) for child in envs_prim.GetChildren() if child.IsActive()]
    else:
        env_paths = ["/World"]

    target_pos = (
        float(goal_origin[0]) + BASKETBALL_LOCAL_OFFSET[0],
        float(goal_origin[1]) + BASKETBALL_LOCAL_OFFSET[1],
        float(goal_z) + BASKETBALL_LOCAL_OFFSET[2],
    )

    spawned = 0
    for env_path in env_paths:
        for fixed_path in (
            f"{env_path}/GoalNet/basketball_semantic",
            f"{env_path}/GoalNet/football_goal/basketball_semantic",
        ):
            fixed_child = stage.GetPrimAtPath(fixed_path)
            if fixed_child and fixed_child.IsValid():
                fixed_child.SetActive(False)

        prim_path = f"{env_path}/SemanticBasketball"
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)

        xform = UsdGeom.Xform.Define(stage, prim_path)
        prim = xform.GetPrim()
        xform.AddTranslateOp().Set(Gf.Vec3d(*target_pos))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))
        prim.GetPayloads().AddPayload(Sdf.Payload(asset_path))

        rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        rigid_api.CreateRigidBodyEnabledAttr(True)
        rigid_api.CreateKinematicEnabledAttr(False)
        rigid_api.CreateStartsAsleepAttr(False)
        rigid_api.CreateVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        rigid_api.CreateAngularVelocityAttr(Gf.Vec3f(0.0, 0.0, 0.0))

        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(DEFAULT_BASKETBALL_MASS)
        spawned += 1

    print(
        f"[semantic_basketball] spawned {spawned} dynamic basketball(s) at "
        f"{target_pos} from {asset_path}"
    )
    return spawned > 0
