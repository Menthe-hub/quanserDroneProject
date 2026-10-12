"""城市配送任务 Demo

用法：
    python Demo/city_mission.py --check              # 体检 + 任务检查（纯离线，秒级）
    python Demo/city_mission.py --selftest           # 自检：证明调度契约的检查真的会拦
    python Demo/city_mission.py --build [--shot P]   # 建图，自由相机看效果（可存截图）
    python Demo/city_mission.py --run [--virtual]    # 让无人机把货送一遍
    python Demo/city_mission.py --seed 7             # 换一个种子重新生成一座城

**地图、任务点、以及要写的算法都在 `Module/StudentMap.py` 里。**
本文件是任务执行器，一般不用改。

## 任务的形状：从"寻路"变成了"调度"

无人机巡航在 **z=8.5**，比所有楼（4~7 m）都高，所以：

* 送货点**可以摆在楼顶上** —— 飞到它正上方就行
* **障碍不构成约束**，两点之间直线可达 —— 寻路本身没有难度了

于是学生要写的就只剩一件事：**`StudentMap.schedule()`**，决定**按什么顺序送**。
每到一个点会打一条提示（取到了货 / 送到了第几个 / 到了终点），不做评分。

⚠ 连着 QLabs 的进程**绝不能用外部 timeout 强杀** —— 客户端异常断开会让 QLabs
  卸载世界。用 `--build --seconds N` 让它自己优雅退出。
"""
import sys
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import Module.QLabsLock  # noqa: F401  必须在任何 qvl 使用之前导入，串行化 QLabs 访问

from Module import MapSpec, StudentMap


def validate_schedule(order_idx, n):
    """检查 `StudentMap.schedule()` 的返回值是不是 0..n-1 的一个排列。

    学生的算法题交的就是这一个函数，所以这个契约必须当场查清楚 ——
    少一个、多一个、越界，都要给出能看懂的错误，而不是让它跑到一半才炸。
    """
    if not isinstance(order_idx, (list, tuple)):
        raise ValueError(f"schedule() 应返回一个列表，实际返回 {type(order_idx).__name__}")
    try:
        idx = [int(x) for x in order_idx]
    except (TypeError, ValueError) as e:
        raise ValueError(f"schedule() 返回的东西没法当索引用：{order_idx!r}（{e}）")
    if sorted(idx) != list(range(n)):
        raise ValueError(
            f"schedule() 返回 {idx}，但它不是 0..{n-1} 的一个排列。\n"
            f"  要求：每个送货点**恰好出现一次**，长度必须是 {n}。\n"
            f"  例：送货点是 {n} 个，返回 [2, 0, 1] 表示按第 3、第 1、第 2 个的顺序送。")
    return idx

# 到位判定。现在是**20~26 米的长直航段**（不再是原来 1 米的逐格小步），
# 桨有惯性、末端会有一点过冲再拉回，所以容差必须比逐格时代放宽。
STEP_TOL = 0.40          # 单站到位判定（米）
GOAL_TOL = 0.45          # 终点到达判定（米）

# 巡航高度取 **> 所有楼**（MapSpec.CRUISE_Z = 8.5，楼是 4~7 m）。两个后果：
#   1) 送货点**可以摆在楼顶** —— 无人机飞到它正上方就行
#   2) **障碍不再构成约束**，两点之间直线可达 —— 寻路本身没有难度了
# 这是刻意的：实验重心从"寻路"挪到"调度"（按什么顺序送），学生的算法题
# 因此变得能自己写完整（见 StudentMap.schedule）。
FLY_Z = MapSpec.CRUISE_Z


# 当前这一轮实际使用的任务点。`--seed` 时会换成自动配的，
# 因为城市是随机生成的，学生文件里写死的坐标必然对不上。
POINTS = {}


def load_student_map(verbose=True, seed=None):
    """把地图装载进 MapSpec —— 之后所有函数都跟着变。

    seed 给了就**用这个种子重新生成一座城**（忽略 StudentMap.ASCII），
    并按新地图**自动配一套任务点**；不给就用学生文件里那份。
    """
    if seed is None:
        MapSpec.load_ascii(StudentMap.ASCII, verbose=verbose)
        POINTS.update(start=tuple(StudentMap.START), pickup=tuple(StudentMap.PICKUP),
                      deliveries=[tuple(d) for d in StudentMap.DELIVERIES],
                      goal=tuple(StudentMap.GOAL))
    else:
        MapSpec.CITY_SEED = seed              # decor / 行人 都跟着这个种子走
        MapSpec.load_ascii(MapSpec.generate_city(seed=seed), verbose=verbose)
        Pickup = (MapSpec.CENTER, MapSpec.CENTER)
        POINTS.update(start=MapSpec.START_CELL, pickup=Pickup,
                      deliveries=MapSpec.auto_deliveries(), goal=MapSpec.GOAL_CELL)
        if verbose:
            print(f"（用种子 {seed} 生成；自动配点：取货 {POINTS['pickup']}，"
                  f"送货 {POINTS['deliveries']}）")
    return POINTS


# ---------------------------------------------------------------------------
# --check：体检 + 任务检查（纯离线）
# ---------------------------------------------------------------------------
def cmd_check(seed=None):
    load_student_map(seed=seed)
    ok, _ = MapSpec.mission_report(POINTS['deliveries'],
                                   start=POINTS['start'],
                                   goal=POINTS['goal'],
                                   pickup=POINTS['pickup'],
                                   flying_over=True)      # 无人机飞在楼之上
    return ok


# ---------------------------------------------------------------------------
# --selftest：证明 schedule() 的契约真的会被检查
# ---------------------------------------------------------------------------
def cmd_selftest(seed=None):
    """自检学生那个 `schedule()` 函数的契约检查是不是真的会拦。

    现在没有"堵路"了（无人机飞在楼之上，直线可达），所以自检的对象从
    "堵路闸门"换成了"调度契约" —— 学生交的就是这一个函数，
    返回的排列必须合法，否则整个任务没法执行。

    全程不写文件、不连 QLabs。
    """
    load_student_map(verbose=False, seed=seed)
    n = len(POINTS['deliveries'])
    print(f"① 学生写的 schedule() 返回值检查（{n} 个送货点）")

    try:
        idx = validate_schedule(StudentMap.schedule(tuple(POINTS['start']),
                                                    tuple(POINTS['pickup']),
                                                    [tuple(d) for d in POINTS['deliveries']],
                                                    tuple(POINTS['goal'])), n)
        print(f"   ✓ 合法：{idx}")
        ok0 = True
    except ValueError as e:
        print(f"   ✗ {e}")
        ok0 = False

    # 故意喂几个非法返回值，确认都会被拦下
    bad_cases = [
        ([0] * n, "缺了别的点、同一个点重复"),
        (list(range(n - 1)), "少了一个点"),
        (list(range(n)) + [0], "多了一个点"),
        (list(range(1, n + 1)), "越界（最大下标是 n-1）"),
    ]
    caught = 0
    print(f"\n② 故意喂非法返回值，应该全被拦下：")
    for bad, why in bad_cases:
        try:
            validate_schedule(bad, n)
            print(f"   ✗ {bad} —— {why}：**没被拦下，闸门失效**")
        except ValueError:
            caught += 1
            print(f"   ✓ {bad} —— {why}：已拦下")
    print(f"   拦截 {caught}/{len(bad_cases)}")

    good = ok0 and caught == len(bad_cases)
    print(f"\n==> 自检 {'✓ 通过：调度契约的检查确实会拦' if good else '✗ 未通过'}")
    return good


# ---------------------------------------------------------------------------
# --build
# ---------------------------------------------------------------------------
def cmd_build(qlabs, seconds, shot=None, pedestrians="dynamic", seed=None):
    import time
    from qvl.free_camera import QLabsFreeCamera

    load_student_map(seed=seed)
    t0 = time.time()
    print(f"开始建「{StudentMap.NAME}」...")
    city = MapSpec.build_scene(qlabs, deliveries=POINTS['deliveries'],
                               pickup=POINTS['pickup'],
                               decor=StudentMap.DECOR, pedestrians=pedestrians,
                               verbose=True)
    print(f"建图完成，用时 {time.time()-t0:.1f}s")

    # 俯视机位**由 SIZE 派生**，否则地图一放大就只能看到一个角。
    #
    # 顺序很讲究（踩过才知道）：spawn -> **set_image_capture_resolution** -> possess
    #   * 不设分辨率，get_image 永远返回 False（SDK 文档写明）
    #   * 分辨率必须在 possess **之前**设，possess 之后再设就不生效了
    # 所以这里先在原点 spawn，设好分辨率、取到相机，最后才 set_transform 挪到俯视位。
    mid = MapSpec.SIZE / 2
    # ⚠ `clean_all()` **不清理自由相机**。上一轮留下的相机会占着 actorNumber=0，
    #    新的 spawn_id(actorNumber=0) 返回 2（编号被占）且不抛异常，
    #    之后所有 set_resolution / get_image 全打在旧相机上。
    QLabsFreeCamera(qlabs).destroy_all_actors_of_class()
    time.sleep(0.3)
    cam = QLabsFreeCamera(qlabs)
    cam.spawn_id(actorNumber=0, location=[0, 0, 0], rotation=[0, 0, 0])
    cam.set_image_capture_resolution(1280, 720)
    cam.possess()
    cam.set_transform([mid, mid, MapSpec.SIZE * 1.55], [0, math.pi/2, 0])

    if shot:
        import cv2
        img = None
        for attempt in range(5):
            time.sleep(1.0)
            ok, img = cam.get_image()
            if ok and img is not None:
                break
            img = None
        if img is not None:
            cv2.imwrite(shot, img)
            print(f"已存俯视截图 -> {shot}  ({img.shape[1]}x{img.shape[0]})")
        else:
            print("截图失败：试了 5 次都拿不到图像")

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
    # 起飞点**跟着当前任务起点走**，不能写死 ——
    # 起点一旦落进楼里（地图改过之后很常见），写死的坐标会让无人机生在墙里，
    # 表现为"起飞失败"，但报错完全看不出来是位置问题。
    si, sj = POINTS['start']
    home = [si + 0.5, sj + 0.5]
    if virtual:
        from Module.VirtualQDrone2 import VirtualQDrone2
        # ⚠ 障碍表给**空**的：虚拟无人机是纯二维的（没有高度概念），
        #   而实体无人机飞在 z=8.5、在楼（4~7 m）之上，客观上没有障碍物，
        #   连"楼顶上有送货点"都是合法的。给它真的墙表反而会误报碰撞。
        d = VirtualQDrone2([], MapSpec.SIZE)
        d.create_qdrone2([home[0], home[1], 0.35], [0, 0, 0], [0.25, 0.25, 0.25])
        d.takeoff([home[0], home[1], FLY_Z])
        return d, "虚拟"
    from Module.QDrone2 import QDrone2
    d = QDrone2(qlabs)
    d.destroy_all_qdrone2()
    if not d.create_qdrone2([home[0], home[1], 0.35], [0, 0, 0], [0.25, 0.25, 0.25]):
        raise SystemExit(f"无人机创建失败（起飞点 {home}）")
    if not d.takeoff([home[0], home[1], FLY_Z]):
        raise SystemExit(f"起飞失败（起飞点 {home}）—— 检查起点是不是落在楼里")
    return d, "真机"


def cmd_run(qlabs, virtual, plot=True, seed=None):
    import time

    load_student_map(verbose=False, seed=seed)
    size = MapSpec.SIZE

    # ---- 1) 规划（先取货，再送货）----
    pickup = tuple(POINTS['pickup'])
    start_c = tuple(POINTS['start'])
    goal_c = tuple(POINTS['goal'])
    deliveries = [tuple(d) for d in POINTS['deliveries']]

    # ---- 2) 让**学生写的调度算法**决定送货顺序 ----
    try:
        order_idx = validate_schedule(StudentMap.schedule(start_c, pickup, deliveries, goal_c),
                                      len(deliveries))
    except ValueError as e:
        raise SystemExit(f"✗ {e}")

    def route(order):
        """按某个配送顺序拼出完整航路：起点 -> 取货 -> 各送货点 -> 终点"""
        return [("起点", start_c), ("取货点", pickup)] \
            + [(f"送货点{i+1}", deliveries[i]) for i in order] \
            + [("终点", goal_c)]

    def length(stops_):
        return sum(math.hypot(b[1][0] - a[1][0], b[1][1] - a[1][1])
                   for a, b in zip(stops_, stops_[1:]))

    stops = route(order_idx)
    total = length(stops)
    naive = length(route(range(len(deliveries))))
    print(f"调度结果：{order_idx}  （学生写的 StudentMap.schedule 决定的送货顺序）")
    print(f"直线总里程 {total:.1f} m" +
          (f"，照文件原顺序送是 {naive:.1f} m（省了 {naive-total:.1f} m）"
           if abs(naive - total) > 0.05 else ""))

    # ---- 3) 建图（真机才需要；而且只建一次）----
    city = None
    if not virtual:
        print("建图中（只建一次）...")
        city = MapSpec.build_scene(qlabs, deliveries=deliveries, pickup=pickup,
                                   decor=StudentMap.DECOR, pedestrians="dynamic",
                                   verbose=False)

    # ---- 4) 起飞并逐点执行，**每到一个点给一条提示** ----
    drone, mode = _make_drone(qlabs, virtual)
    print(f"\n{mode}起飞于 {start_c}，巡航高度 {FLY_Z} m（在楼顶之上）")

    visited, log = set(), []
    got_pickup = [False]
    xs, ys = [start_c[0] + 0.5], [start_c[1] + 0.5]
    t0 = time.time()
    abort = None

    for k in range(1, len(stops)):
        name, (ci, cj) = stops[k]
        prev = stops[k - 1][1]
        wp = [ci + 0.5, cj + 0.5, FLY_Z]
        hop = math.hypot(ci - prev[0], cj - prev[1])
        print(f"  → 前往{name} ({ci}, {cj})   直线 {hop:5.1f} m", flush=True)

        ok, loc, coll = drone.fly_to(wp)
        xs.append(loc[0]); ys.append(loc[1])

        if coll:
            abort = f"前往{name}时撞到东西，位置 ({loc[0]:.2f}, {loc[1]:.2f})"
            break
        if math.dist(loc[:2], wp[:2]) > STEP_TOL:
            abort = f"没飞到{name}（离目标 {math.dist(loc[:2], wp[:2]):.3f} m）"
            break

        here = MapSpec.roof_top(ci, cj)
        where = "（楼顶）" if here > 0 else ""
        if name == "取货点":
            got_pickup[0] = True
            print(f"    ✓ 已取货 {where}", flush=True)
        elif name.startswith("送货点"):
            idx = order_idx.index(int(name[3:]) - 1)
            visited.add((ci, cj))
            print(f"    ✓ 送达 {idx+1}/{len(deliveries)} {where}", flush=True)
        elif name == "终点":
            print(f"    ✓ 抵达终点", flush=True)
        log.append(name)
    el = time.time() - t0

    # ---- 5) 判定（不做评分，只要能到就行）----
    print()
    end = [xs[-1], ys[-1]]
    reached_goal = math.dist(end, [goal_c[0] + 0.5, goal_c[1] + 0.5]) < GOAL_TOL
    missed = set(deliveries) - visited

    if abort:
        print(f"✗ 任务中止：{abort}")
    elif not reached_goal:
        print(f"✗ 没到终点：最后位置 ({end[0]:.2f}, {end[1]:.2f})")
    elif not got_pickup[0]:
        print(f"✗ 没到取货点 {pickup}")
    elif missed:
        print(f"✗ 有送货点没送到：{sorted(missed)}")
    else:
        print(f"✓ 任务完成：{el:.1f}s，取货 1 处 + 送达 {len(visited)}/{len(deliveries)} 处，"
              f"无碰撞，直线总里程 {total:.1f} m")
        print(f"  实际经过：{' -> '.join(log)}")

    if city is not None:
        city.stop_pedestrians()

    # ---- 5) 轨迹图（用 Draw，不做评分）----
    #    注意：Drawing.draw() 结尾是 plt.show()，**会阻塞到窗口关闭**。
    #    自动化跑（无人值守）请加 --no-plot。
    if not plot:
        return abort is None and reached_goal and got_pickup[0] and not missed
    try:
        from Module.Drawing import Draw
        Draw(size, MapSpec.occupancy()).draw(xs, ys, f"Delivery Mission - {StudentMap.NAME}",
                                             "x", "y", show_MAZE=True, trajectory=True)
    except Exception as e:
        print(f"(轨迹图绘制跳过: {e})")
    return abort is None and reached_goal and got_pickup[0] and not missed


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="城市配送任务")
    ap.add_argument("--check", action="store_true", help="体检 + 任务检查（纯离线）")
    ap.add_argument("--selftest", action="store_true", help="自检：证明调度契约的检查真的会拦")
    ap.add_argument("--build", action="store_true", help="建图并用相机看效果")
    ap.add_argument("--run", action="store_true", help="执行送货任务")
    ap.add_argument("--virtual", action="store_true", help="配合 --run：用纯网格虚拟无人机，不连 QLabs")
    ap.add_argument("--no-plot", action="store_true", help="配合 --run：不画轨迹图（plt.show 会阻塞到窗口关闭）")
    ap.add_argument("--seconds", type=int, default=0, help="配合 --build：N 秒后自动结束")
    ap.add_argument("--shot", metavar="PATH", help="配合 --build：存一张俯视截图")
    ap.add_argument("--seed", type=int, default=None,
                    help="用这个种子重新生成一座城（不填就用 StudentMap 里那份）")
    ap.add_argument("--no-pedestrians", action="store_true", help="配合 --build：不放行人")
    a = ap.parse_args()

    if a.check:
        sys.exit(0 if cmd_check(a.seed) else 1)
    if a.selftest:
        sys.exit(0 if cmd_selftest(a.seed) else 1)

    if a.run and a.virtual:
        sys.exit(0 if cmd_run(None, True, plot=not a.no_plot, seed=a.seed) else 1)

    from qvl.qlabs import QuanserInteractiveLabs
    qlabs = QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        raise SystemExit("无法连接 QLabs")
    try:
        if a.build:
            cmd_build(qlabs, a.seconds, shot=a.shot,
                      pedestrians="off" if a.no_pedestrians else "dynamic", seed=a.seed)
        elif a.run:
            cmd_run(qlabs, False, plot=not a.no_plot, seed=a.seed)
        else:
            ap.print_help()
    finally:
        qlabs.close()
