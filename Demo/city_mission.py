"""城市配送任务 Demo

用法：
    python Demo/city_mission.py --check              # 体检 + 堵路测试（纯离线，秒级）
    python Demo/city_mission.py --selftest           # 自检：证明堵路测试真的会拦
    python Demo/city_mission.py --build [--seconds N]# 建图，自由相机看效果
    python Demo/city_mission.py --run [--virtual]    # 让无人机把货送一遍

**地图和任务都在 `Module/StudentMap.py` 里** —— 那个文件才是要改的地方。
本文件是任务执行器，一般不用改。

任务流程：规划 = A* 分段规划 START -> DELIVERIES[0] -> ... -> GOAL；
          执行 = 无人机沿规划出的格子路径逐格飞；碰撞或没到位就中止。

⚠ 连着 QLabs 的进程**绝不能用外部 timeout 强杀** —— 客户端异常断开会让 QLabs
  卸载世界。用 `--build --seconds N` 让它自己优雅退出。
"""
import sys
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import Module.QLabsLock  # noqa: F401  必须在任何 qvl 使用之前导入，串行化 QLabs 访问

from Module import MapSpec, Astar, StudentMap

STEP_TOL = 0.15          # 单步到位判定（米）
GOAL_TOL = 0.30          # 终点到达判定（米）
FLY_Z = 1.0


def load_student_map(verbose=True):
    """把 StudentMap.ASCII 装载进 MapSpec —— 之后所有函数都跟着变"""
    MapSpec.load_ascii(StudentMap.ASCII, verbose=verbose)


# ---------------------------------------------------------------------------
# --check：体检 + 堵路测试
# ---------------------------------------------------------------------------
def cmd_check():
    load_student_map()
    ok, _ = MapSpec.mission_report(StudentMap.DELIVERIES,
                                   start=StudentMap.START,
                                   goal=StudentMap.GOAL)
    return ok


# ---------------------------------------------------------------------------
# --selftest：证明堵路测试真的会拦
# ---------------------------------------------------------------------------
def cmd_selftest():
    """在内存里把某个投递点围死，确认验收会失败并指出是哪一段；再恢复。

    全程不写文件、不连 QLabs。
    """
    load_student_map(verbose=False)
    ok0, _ = MapSpec.mission_report(StudentMap.DELIVERIES, start=StudentMap.START,
                                    goal=StudentMap.GOAL, verbose=False)
    print(f"① 原图验收            {'✓ 通过' if ok0 else '✗ 不通过'}")

    # 挑一个投递点，用墙把它四周围死
    ti, tj = StudentMap.DELIVERIES[0]
    rows = list(StudentMap.ASCII)
    def put(i, j):
        r = list(rows[j]); r[i] = "#"; rows[j] = "".join(r)
    put(ti, tj)                                     # 本体也封上
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        i, j = ti + di, tj + dj
        if 0 <= i < MapSpec.SIZE and 0 <= j < MapSpec.SIZE:
            put(i, j)
    MapSpec.load_ascii(rows, verbose=False)
    print(f"\n② 把投递点 {StudentMap.DELIVERIES[0]} 用墙围死后再体检：")
    ok1, _ = MapSpec.mission_report(StudentMap.DELIVERIES, start=StudentMap.START,
                                    goal=StudentMap.GOAL, verbose=True)
    print(f"   -> {'✗ 闸门失效' if ok1 else '✓ 闸门生效（验收不通过）'}")

    MapSpec.load_ascii(StudentMap.ASCII, verbose=False)
    ok2, _ = MapSpec.mission_report(StudentMap.DELIVERIES, start=StudentMap.START,
                                    goal=StudentMap.GOAL, verbose=False)
    print(f"\n③ 恢复原图后         {'✓ 通过（恢复正常）' if ok2 else '✗ 恢复失败'}")

    good = (ok0 and not ok1 and ok2)
    print(f"\n==> 自检 {'✓ 通过：堵路测试确实会拦' if good else '✗ 未通过'}")
    return good


# ---------------------------------------------------------------------------
# --build
# ---------------------------------------------------------------------------
def cmd_build(qlabs, seconds):
    import time
    from qvl.free_camera import QLabsFreeCamera

    load_student_map()
    t0 = time.time()
    print(f"开始建「{StudentMap.NAME}」...")
    city = MapSpec.build_scene(qlabs, deliveries=StudentMap.DELIVERIES,
                               decor=StudentMap.DECOR, verbose=True)
    print(f"建图完成，用时 {time.time()-t0:.1f}s")

    cam = QLabsFreeCamera(qlabs)
    cam.spawn_id(actorNumber=0, location=[9.5, 9.5, 22.0], rotation=[0, math.pi/2, 0])
    cam.possess()
    try:
        if seconds:
            print(f"已切到俯视相机 —— {seconds} 秒后自动结束")
            time.sleep(seconds)
        else:
            print("已切到俯视相机 —— Ctrl+C 结束")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C")
    finally:
        city.stop_pedestrians()
        print("正常退出")
    return city


# ---------------------------------------------------------------------------
# --run：执行送货任务
# ---------------------------------------------------------------------------
def _make_drone(qlabs, virtual):
    if virtual:
        from Module.VirtualQDrone2 import VirtualQDrone2
        d = VirtualQDrone2(MapSpec.wall_points(), MapSpec.SIZE)
        d.create_qdrone2([0.5, 0.5, 0.35], [0, 0, 0], [0.25, 0.25, 0.25])
        d.takeoff([0.5, 0.5, FLY_Z])
        return d, "虚拟"
    from Module.QDrone2 import QDrone2
    d = QDrone2(qlabs)
    d.destroy_all_qdrone2()
    if not d.create_qdrone2([0.5, 0.5, 0.35], [0, 0, 0], [0.25, 0.25, 0.25]):
        raise SystemExit("无人机创建失败")
    if not d.takeoff([0.5, 0.5, FLY_Z]):
        raise SystemExit("起飞失败")
    return d, "真机"


def cmd_run(qlabs, virtual):
    import time

    load_student_map(verbose=False)
    size = MapSpec.SIZE

    # ---- 1) 规划 ----
    cells, legs = Astar.plan_route(StudentMap.START, StudentMap.DELIVERIES,
                                   goal=StudentMap.GOAL)
    if cells is None:
        bad = next((L for L in legs if not L["ok"]), None)
        raise SystemExit(f"✗ 规划失败：第 {legs.index(bad)+1} 段 {bad['from']} -> {bad['to']} 被堵死。"
                         f"先跑 --check 看看。")
    print(f"规划完成：{len(cells)-1} 步，{len(legs)} 段")
    for k, L in enumerate(legs, 1):
        print(f"  第{k}段 {L['from']} -> {L['to']}  {L['steps']} 步")
    waypoints = Astar.path_to_world(cells, z=FLY_Z)

    # ---- 2) 建图（真机才需要；而且只建一次）----
    city = None
    if not virtual:
        print("建图中（只建一次）...")
        city = MapSpec.build_scene(qlabs, deliveries=StudentMap.DELIVERIES,
                                   decor=StudentMap.DECOR, verbose=False)

    # ---- 3) 起飞并执行 ----
    drone, mode = _make_drone(qlabs, virtual)
    print(f"\n{mode}执行任务：{StudentMap.START} -> {StudentMap.DELIVERIES} -> {StudentMap.GOAL}")

    targets = set(tuple(d) for d in StudentMap.DELIVERIES)
    visited = set()
    xs, ys = [waypoints[0][0]], [waypoints[0][1]]
    t0 = time.time()
    abort = None
    k = 0
    for k, wp in enumerate(waypoints[1:], 1):
        ok, loc, coll = drone.fly_to(wp)
        xs.append(loc[0]); ys.append(loc[1])
        cell = (int(round(loc[0] - 0.5)), int(round(loc[1] - 0.5)))
        if cell in targets:
            visited.add(cell)
        if coll:
            abort = f"第 {k} 步撞上障碍物，位置 ({loc[0]:.2f}, {loc[1]:.2f})"
            break
        if math.dist(loc[:2], wp[:2]) > STEP_TOL:
            abort = f"第 {k} 步没到位（离目标 {math.dist(loc[:2], wp[:2]):.3f} m）"
            break
    el = time.time() - t0

    # ---- 4) 判定（不做评分，只要能到就行）----
    print()
    end = [xs[-1], ys[-1]]
    reached_goal = math.dist(end, [StudentMap.GOAL[0] + 0.5, StudentMap.GOAL[1] + 0.5]) < GOAL_TOL
    missed = targets - visited

    if abort:
        print(f"✗ 任务中止：{abort}")
    elif not reached_goal:
        print(f"✗ 没到终点：最后位置 ({end[0]:.2f}, {end[1]:.2f})")
    elif missed:
        print(f"✗ 有投递点没经过：{sorted(missed)}")
    else:
        print(f"✓ 任务完成：{len(cells)-1} 步，{el:.1f}s，"
              f"{len(visited)}/{len(targets)} 个投递点全部送达，无碰撞")

    if city is not None:
        city.stop_pedestrians()

    # ---- 5) 轨迹图（用 Draw，不做评分）----
    try:
        from Module.Drawing import Draw
        Draw(size, MapSpec.occupancy()).draw(xs, ys, f"Delivery Mission - {StudentMap.NAME}",
                                             "x", "y", show_MAZE=True, trajectory=True)
    except Exception as e:
        print(f"(轨迹图绘制跳过: {e})")
    return abort is None and reached_goal and not missed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="城市配送任务")
    ap.add_argument("--check", action="store_true", help="体检 + 堵路测试（纯离线）")
    ap.add_argument("--selftest", action="store_true", help="自检：证明堵路测试真的会拦")
    ap.add_argument("--build", action="store_true", help="建图并用相机看效果")
    ap.add_argument("--run", action="store_true", help="执行送货任务")
    ap.add_argument("--virtual", action="store_true", help="配合 --run：用纯网格虚拟无人机，不连 QLabs")
    ap.add_argument("--seconds", type=int, default=0, help="配合 --build：N 秒后自动结束")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if cmd_check() else 1)
    if a.selftest:
        sys.exit(0 if cmd_selftest() else 1)

    if a.run and a.virtual:
        sys.exit(0 if cmd_run(None, True) else 1)

    from qvl.qlabs import QuanserInteractiveLabs
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        raise SystemExit("无法连接 QLabs")
    try:
        if a.build:
            cmd_build(qlabs, a.seconds)
        elif a.run:
            cmd_run(qlabs, False)
        else:
            ap.print_help()
    finally:
        qlabs.close()
