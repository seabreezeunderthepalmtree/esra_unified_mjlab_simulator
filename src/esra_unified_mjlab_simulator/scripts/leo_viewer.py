"""View Leo with asset constants; open http://localhost:8080 for joint controls."""

import re
import time

import mujoco
import numpy as np
import viser

from esra_unified_mjlab_simulator.assets.leo_quadruped.leo_constants import (
    HIP_ACTUATOR,
    INIT_STATE,
    KNEE_ACTUATOR,
    LEO_HIP_ACTUATOR_CFG,
    LEO_KNEE_ACTUATOR_CFG,
    get_spec,
)


def main():
    spec = get_spec()  # XML already includes a rigid floor.
    for motor in spec.actuators:
        joint = spec.joint(motor.target)
        actuator = KNEE_ACTUATOR if joint.name.endswith("knee_pitch") else HIP_ACTUATOR
        limit = actuator.effort_limit
        motor.ctrlrange = [-limit, limit]
        motor.ctrllimited = True
        joint.actfrcrange = [-limit, limit]
        joint.actfrclimited = True
        joint.armature = actuator.reflected_inertia
    model = spec.compile()
    default_gravity = model.opt.gravity.copy()
    data = mujoco.MjData(model)
    joints = model.actuator_trnid[:, 0]
    qadr, vadr = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    gains = [
        LEO_KNEE_ACTUATOR_CFG
        if model.joint(int(j)).name.endswith("knee_pitch")
        else LEO_HIP_ACTUATOR_CFG
        for j in joints
    ]
    kp = np.array([cfg.stiffness for cfg in gains])
    kd = np.array([cfg.damping for cfg in gains])
    server = viser.ViserServer(host="127.0.0.1")
    server.scene.set_up_direction("+z")
    server.scene.add_grid("/floor", plane="xy", infinite_grid=True, plane_opacity=1.0)

    paused = server.gui.add_checkbox("Pause", initial_value=True)
    gravity = server.gui.add_checkbox("Gravity", initial_value=True)
    mode = server.gui.add_dropdown("Control", options=("Torque", "Angle"))
    reset = server.gui.add_button("Reset")
    zero = server.gui.add_button("Zero torques")
    home = server.gui.add_button("Default angles")
    # Keep direct torque control; limits and inertia come from leo_constants.py.
    with server.gui.add_folder("Joint torques (Nm)"):
        sliders = [
            server.gui.add_slider(
                f"{model.actuator(i).name.removesuffix('_ctrl')} "
                f"[{model.actuator_ctrlrange[i, 0]:g}, "
                f"{model.actuator_ctrlrange[i, 1]:g}] Nm",
                min=float(model.actuator_ctrlrange[i, 0]),
                max=float(model.actuator_ctrlrange[i, 1]),
                step=0.1,
                initial_value=0.0,
            )
            for i in range(model.nu)
        ]

    with server.gui.add_folder("Joint angles (deg)"):
        angles = [
            server.gui.add_slider(
                model.joint(int(j)).name,
                min=float(np.rad2deg(model.jnt_range[j, 0])),
                max=float(np.rad2deg(model.jnt_range[j, 1])),
                step=0.1,
                initial_value=0.0,
            )
            for j in joints
        ]

    bodies = []
    with server.gui.add_folder("Show body (visual only)"):
        for b in range(1, model.nbody):
            frame = server.scene.add_frame(f"/robot/{b}", show_axes=False)
            toggle = server.gui.add_checkbox(model.body(b).name, initial_value=True)
            bodies.append((b, frame, toggle))
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != b or model.geom_group[g] != 2:
                    continue
                material = model.geom_matid[g]
                rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[g]
                kwargs = dict(
                    color=tuple(int(c * 255) for c in rgba[:3]),
                    position=model.geom_pos[g],
                    wxyz=model.geom_quat[g],
                )
                name = f"/robot/{b}/{g}"
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                    m = model.geom_dataid[g]
                    v, f = model.mesh_vertadr[m], model.mesh_faceadr[m]
                    server.scene.add_mesh_simple(
                        name,
                        model.mesh_vert[v : v + model.mesh_vertnum[m]],
                        model.mesh_face[f : f + model.mesh_facenum[m]],
                        **kwargs,
                    )
                elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    server.scene.add_icosphere(
                        name, radius=model.geom_size[g, 0], **kwargs
                    )

    @server.on_client_connect
    def connected(client):
        client.camera.position = (1.2, -1.2, 0.8)
        client.camera.look_at = (0.0, 0.0, 0.25)

    def reset_state():
        mujoco.mj_resetData(model, data)
        data.qpos[:3] = INIT_STATE.pos
        data.qpos[3:7] = INIT_STATE.rot
        data.qvel[:3] = INIT_STATE.lin_vel
        data.qvel[3:6] = INIT_STATE.ang_vel
        for j in range(1, model.njnt):
            name = model.joint(j).name
            for pattern, value in (INIT_STATE.joint_pos or {}).items():
                if re.fullmatch(pattern, name):
                    data.qpos[model.jnt_qposadr[j]] = value
            for pattern, value in INIT_STATE.joint_vel.items():
                if re.fullmatch(pattern, name):
                    data.qvel[model.jnt_dofadr[j]] = value
        for slider in sliders:
            slider.value = 0.0
        for slider, value in zip(angles, data.qpos[qadr], strict=True):
            slider.value = float(np.rad2deg(value))
        mujoco.mj_forward(model, data)

    reset_state()
    default_angles = [slider.value for slider in angles]
    last_reset, last_zero = reset.update_timestamp, zero.update_timestamp
    last_home = home.update_timestamp
    try:
        while True:
            start = time.monotonic()
            # Button events are consumed here so only this thread changes physics.
            if reset.update_timestamp > last_reset:
                reset_state()
                last_reset = reset.update_timestamp
            if zero.update_timestamp > last_zero:
                for slider in sliders:
                    slider.value = 0.0
                mode.value = "Torque"
                last_zero = zero.update_timestamp
            if home.update_timestamp > last_home:
                for slider, value in zip(angles, default_angles, strict=True):
                    slider.value = value
                for slider in sliders:
                    slider.value = 0.0
                mode.value = "Angle"
                last_home = home.update_timestamp
            model.opt.gravity[:] = default_gravity if gravity.value else 0.0
            target = np.deg2rad([slider.value for slider in angles])
            if paused.value and mode.value == "Angle":
                # Pose editing without advancing simulation time.
                data.qpos[qadr] = target
                data.qvel[vadr] = 0.0
                mujoco.mj_forward(model, data)
            elif not paused.value:
                # Recompute PD torque each physics step using the asset gains.
                for _ in range(10):
                    torque = (
                        kp * (target - data.qpos[qadr]) - kd * data.qvel[vadr]
                        if mode.value == "Angle"
                        else np.array([slider.value for slider in sliders])
                    )
                    data.ctrl[:] = np.clip(
                        torque,
                        model.actuator_ctrlrange[:, 0],
                        model.actuator_ctrlrange[:, 1],
                    )
                    mujoco.mj_step(model, data)
            with server.atomic():
                for b, frame, toggle in bodies:
                    frame.position = data.xpos[b]
                    frame.wxyz = data.xquat[b]
                    frame.visible = toggle.value
            time.sleep(max(0.0, 10 * model.opt.timestep - (time.monotonic() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
