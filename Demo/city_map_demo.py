"""示例地图 Demo（阶段 2）

展示 Module/MapSpec.py 这套"ASCII 地图 -> QLabs 场景 + 占据栅格"的用法。

用法：
    python Demo/city_map_demo.py --check                     # 纯离线自检，不连 QLabs
    python Demo/city_map_demo.py --build                     # 建图，肉眼验收
    python Demo/city_map_demo.py --build --pedestrians dynamic
    python Demo/city_map_demo.py --run --virtual             # 用新地图跑一遍 Q-Learning 训练+测试

**不修改任何现有文件。** 跑算法时通过覆盖 QLearningSimulation 的模块级全局变量
（SIZE / MAZE / MAZE_POINTS / ORIGINE / GOAL）来复用它整套训练、奖励、绘图、
评估流程 —— 这个做法在项目里的多次实验中已经反复验证过。
"""
import sys, argparse, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 必须在任何 qvl 使用之前导入：把 QLabs 的访问串行化。
# 不导的话，行人线程与主线程会互相吃掉对方的回包（实测：无人机在第 2 步就误报碰撞）。
import Module.QLabsLock  # noqa: F401

from Module import MapSpec


def cmd_check():
    """纯离线：不连 QLabs 也能跑，用来在改地图后立刻发现问题"""
    steps = MapSpec.report()
    print()
    print("  提示：最短路径 = 曼哈顿下界说明地图很开放（对 A* 是平凡解、对 RL 较容易）。")
    print("        想加难度就编辑 Module/MapSpec.py 里的 BASE_ASCII，再跑一次 --check。")


def cmd_build(qlabs, pedestrians, seconds):
    import time
    from qvl.free_camera import QLabsFreeCamera

    t0 = time.time()
    print(f"开始建图（障碍 {len(MapSpec.wall_cells())} 格，行人模式 {pedestrians}）...")
    city = MapSpec.build_scene(qlabs, pedestrians=pedestrians)
    print(f"建图完成，用时 {time.time()-t0:.1f}s")

    cam = QLabsFreeCamera(qlabs)
    cam.spawn_id(actorNumber=0, location=[9.5, -12.0, 20.0], rotation=[0, math.radians(60), 0])
    cam.possess()
    try:
        if seconds:
            print(f"已切到自由相机（俯视）—— {seconds} 秒后自动结束（不再需要强杀）")
            time.sleep(seconds)
        else:
            print("已切到自由相机（俯视）—— Ctrl+C 结束")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，准备收尾。")
    finally:
        city.stop_pedestrians()
        print("已停止所有行人线程，正常退出")


def cmd_run(qlabs, virtual, episodes, pedestrians):
    """用新地图跑一遍训练 + 测试"""
    import numpy as np
    import Demo.QLearningSimulation as S
    from qvl.qlabs import QuanserInteractiveLabs

    # ---- 把新地图灌进现有 Demo 的模块级全局变量 ----
    S.SIZE = MapSpec.SIZE
    S.MAZE_POINTS = MapSpec.wall_points()
    S.MAZE = MapSpec.occupancy()
    S.ORIGINE = MapSpec.ORIGINE
    S.GOAL = MapSpec.GOAL
    if episodes:
        S.EPISODES = episodes
    print(f"已把新地图灌入 QLearningSimulation：障碍 {len(S.MAZE_POINTS)} 格，"
          f"最短路径 {MapSpec.shortest_path()[0]} 步，EPISODES={S.EPISODES}")

    if virtual:
        S.VIRTUAL = True
        print("虚拟模式：不连 QLabs，也不建图")
    else:
        S.VIRTUAL = False
        print("真机模式：建图中...")
        city = MapSpec.build_scene(qlabs, pedestrians=pedestrians)

    drawer = S.Draw(S.SIZE, S.MAZE)
    agent = S.qlearning_train(qlabs, drawer)

    ev = S.Evaluator()
    x, y, finished = S.qlearning_test(qlabs, agent, ev)
    drawer.draw(x, y, "Trajectory (city_map_demo)", "x", "y")
    print(ev.generate_evaluation_report(finished))
    ev.plot_performance_analysis()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="示例地图 Demo")
    ap.add_argument("--check", action="store_true", help="只跑离线自检，不连 QLabs")
    ap.add_argument("--build", action="store_true", help="建图并用自由相机看效果")
    ap.add_argument("--run", action="store_true", help="建图并跑一遍训练+测试")
    ap.add_argument("--pedestrians", choices=["off", "dynamic"], default="off",
                    help="off=不放行人；dynamic=行人在街道上巡逻（会挡路，环境变非确定）")
    ap.add_argument("--virtual", action="store_true", help="配合 --run：用纯网格虚拟环境训练")
    ap.add_argument("--episodes", type=int, default=None, help="覆盖 EPISODES")
    ap.add_argument("--seconds", type=int, default=0,
                    help="配合 --build：N 秒后自动结束（0=一直等到 Ctrl+C）。"
                         "注意别用外部 timeout 强杀连着 QLabs 的进程，会让 QLabs 卸载世界")
    a = ap.parse_args()

    if a.check:
        cmd_check()
        sys.exit(0)

    from qvl.qlabs import QuanserInteractiveLabs
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        raise SystemExit("无法连接 QLabs")

    if a.build:
        cmd_build(qlabs, a.pedestrians, a.seconds)
    elif a.run:
        cmd_run(qlabs, a.virtual, a.episodes, a.pedestrians)
    else:
        ap.print_help()
