"""Interactive visual demo of Stage 2 Kinematics and Pick-and-Place trajectory in MuJoCo."""

from __future__ import annotations
import time
import numpy as np
import mujoco
import mujoco.viewer

from src.simulation.mujoco_sim import MujocoSim
from src.kinematics.solver import KinematicsSolver
from src.kinematics.trajectory import TrajectoryGenerator


def run_kinematics_demo(headless: bool = False) -> None:
    print("=" * 70)
    print("Manipulator2: Интерактивная демонстрация кинематики и траекторий (Этап 2)")
    print("=" * 70)

    # 1. Initialize Simulator, Solver, and Generator
    sim = MujocoSim()
    solver = KinematicsSolver()
    generator = TrajectoryGenerator()

    # Get red box position on the table
    target_pos = sim.get_object_position("red_box")
    # Grasp target slightly above bottom center of cube
    grasp_pos = np.array([target_pos[0], target_pos[1], 0.025])
    dropoff_pos = np.array([0.18, 0.20, 0.035])

    print(f"\n1. Целевой объект: 'red_box' на координатах: {np.round(grasp_pos, 3)}")
    print(f"2. Зона сброса (Drop-off): {np.round(dropoff_pos, 3)}")

    # 2. Plan full pick and place pipeline
    print("\n3. Расчет обратной кинематики (IK) и S-образной траектории...")
    waypoints, trajectory = generator.plan_pick_and_place(
        pick_position=grasp_pos,
        dropoff_position=dropoff_pos,
        solver=solver,
    )

    print(f"   Сгенерировано контрольных точек (Waypoints): {len(waypoints)}")
    for i, wp in enumerate(waypoints):
        print(f"     [{i+1}] {wp.name:12s} -> Схват: {'ОТКРЫТ' if wp.gripper > 0.02 else 'ЗАКРЫТ':7s} | {wp.description}")
    print(f"   Общее количество точек траектории: {len(trajectory)}")
    print(f"   Общее расчетное время цикла: {trajectory[-1].time:.2f} сек")

    # 3. Execution in MuJoCo
    print("\n4. Запуск симуляции физического выполнения...")

    if not headless:
        print("   Открытие 3D-окна MuJoCo Viewer... (нажмите ESC или закройте окно для выхода)")
        try:
            with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
                viewer.cam.distance = 1.2
                viewer.cam.elevation = -25
                viewer.cam.azimuth = 135
                viewer.cam.lookat = [0.25, 0.05, 0.15]

                dt_sim = sim.model.opt.timestep
                steps_per_pt = max(1, int(round(0.01 / dt_sim)))

                current_stage = ""
                for pt in trajectory:
                    if not viewer.is_running():
                        break

                    if pt.segment_name != current_stage:
                        current_stage = pt.segment_name
                        print(f"   >>> Этап движения: {current_stage}")

                    sim.set_joint_targets(pt.positions)
                    sim.set_gripper(pt.gripper)

                    # Physical grasping interaction
                    if "GRASP" in pt.segment_name and pt.gripper < 0.02:
                        sim.grasp_object("red_box")
                    elif "PLACE" in pt.segment_name and pt.gripper > 0.02:
                        sim.release_object()

                    sim.step(steps_per_pt)
                    viewer.sync()
                    time.sleep(0.01)

                print("\n   Цикл успешно завершен! Удержание позы готовности...")
                for _ in range(150):
                    if not viewer.is_running():
                        break
                    sim.step(1)
                    viewer.sync()
                    time.sleep(0.01)
        except Exception as e:
            print(f"   Примечание: GUI viewer недоступен ({e}), выполняется в headless-режиме.")
            headless = True

    if headless:
        dt_sim = sim.model.opt.timestep
        steps_per_pt = max(1, int(round(0.01 / dt_sim)))
        for pt in trajectory:
            sim.set_joint_targets(pt.positions)
            sim.set_gripper(pt.gripper)
            if "GRASP" in pt.segment_name and pt.gripper < 0.02:
                sim.grasp_object("red_box")
            elif "PLACE" in pt.segment_name and pt.gripper > 0.02:
                sim.release_object()
            sim.step(steps_per_pt)
        print("   Headless-симуляция успешно выполнена!")

    final_tcp = sim.data.site_xpos[sim.tcp_site_id]
    print(f"\nИтоговые координаты TCP в симуляторе: {np.round(final_tcp, 3)}")
    print("=" * 70)
    sim.close()


if __name__ == "__main__":
    run_kinematics_demo(headless=False)
