"""地图定义与编译器（阶段 2）

与现有 Demo 完全解耦：不 import QLearningSimulation，也不改动它。
现有两个 Demo 一行未变，本模块是新增的独立地图来源。

## 三层结构

    走廊格 / 障碍格（ASCII）──► compile() ──► wall_points() / occupancy()
                                          └─► build_scene()   生成 QLabs 场景
    装饰层 ─────────────────────────► decorate()   只生成 actor，**永不写入 occupancy**

不变量：`occupancy()` 只由 ASCII 决定，装饰层没有任何路径能影响它。
所以"加装饰不破坏算法"是结构上成立的，不是靠约定。

## 由探针实测得出的约束（见 probe/阶段0-结论.md）

* 无人机飞在 z=1.0；道具**高于约 0.6 会挡路，矮于 0.6 不挡**
  → 矮装饰（路面、车道线、人行横道）可以铺在走廊上
  → 高装饰（树）只能坐落在**墙格**上——渲染成树而不是楼，碰撞行为完全一样
* **行人永远挡路**，没有任何接口能关掉（`enable_collsion` 只管计数）
  → 两种模式只能是「有行人（挡路、非确定）」和「无行人」，做不到"不挡路的景观行人"
* `QLabsWalls`（泡沫板墙）无法做成障碍，是纯视觉组件
* 建图 **66.7 ms/块**，一张 19×19 约 16 秒 → **只能建一次**，绝不能放进 reset_environment
"""

import random

import numpy as np

# 把 QLabs 的访问串行化（幂等）。只在真正连 QLabs 时才起作用，虚拟模式无影响。
# 不加的话：用了行人线程（dynamic 模式）后，主线程的请求会被线程抢走应答，
# 表现为无人机误报碰撞 / spawn 静默失败。详见 Module/QLabsLock.py。
from Module import QLabsLock  # noqa: F401

SIZE = 19
ORIGINE = [0.5, 0.5]
GOAL = [SIZE - 0.5, SIZE - 0.5]

# ---------------------------------------------------------------------------
# 地图本体：19x19，每行 19 个字符
#   '.' 可通行     '#' 建筑     'T' 树（与 '#' 同为障碍，仅外观不同）
# 骨架是"每 3 格一条街"的城市网格，再手工加了几个街区变化
# ---------------------------------------------------------------------------
BASE_ASCII = [
    "...................",
    ".##.##.##.##.##.##.",
    ".##.##.##.##.##.##.",
    "...................",
    ".##.##.##.##.##.##.",
    ".##.##.##.##.##.##.",
    "...................",
    ".##.##.##.##.##.##.",
    ".##.##.##.##.##.##.",
    ".........###.......",
    ".##.##.####.##.##..",
    ".##.##.####.##.##..",
    "....#..............",
    ".##.##.##.##.##.##.",
    ".##.##.##.##.##.##.",
    "...................",
    ".##.##.##.##.##.##.",
    ".##.##.TT.##.##.##.",
    "...................",
]

assert len(BASE_ASCII) == SIZE, f"ASCII 行数应为 {SIZE}"
for _r, _row in enumerate(BASE_ASCII):
    assert len(_row) == SIZE, f"第 {_r} 行长度 {len(_row)}，应为 {SIZE}"

# ---- 外观与合法性常量 ----

VALID_CHARS = {".", "#", "T"}          # 学生能用的字符集

# 楼的外墙配色。刻意选接近的灰调，让城市有层次而不是花哨。
BUILDING_PALETTE = [
    [0.30, 0.30, 0.30],
    [0.35, 0.32, 0.30],
    [0.28, 0.30, 0.34],
    [0.33, 0.31, 0.28],
    [0.30, 0.33, 0.33],
    [0.36, 0.34, 0.32],
]
DELIVERY_COLOR = [1.0, 0.55, 0.0]      # 投递点地砖：橙色（场景里没有文字标签，只能靠颜色）


def building_style(i, j):
    """按格子坐标**确定性**地给出这栋楼的高度与颜色。

    规则式而非逐格数据 —— 学生改完 ASCII 后自动跟随，不用手工维护。
    占地仍然是 1x1 格，所以**碰撞行为逐格不变**，只是整座城市不再是一排
    一模一样的灰方块。
    """
    h = 2.0 + ((i * 7 + j * 13) % 4)                       # 2.0 / 3.0 / 4.0 / 5.0
    color = BUILDING_PALETTE[(i * 5 + j * 11) % len(BUILDING_PALETTE)]
    return h, color


# 街区行/列的判定：每 3 格一条街
def is_street(i, j):
    """该格是不是街道（用于铺路面）"""
    return i % 3 == 0 or j % 3 == 0


# 动态模式下行人走的路线（世界坐标 [x, y]，沿街道）
PEDESTRIAN_ROUTES = [
    ([3.5, 0.5], [3.5, 9.5]),
    ([15.5, 18.5], [15.5, 9.5]),
    ([0.5, 15.5], [12.5, 15.5]),
]


# ---------------------------------------------------------------------------
# 编译：ASCII -> wall_points / occupancy
# ---------------------------------------------------------------------------

def _cell_char(i, j):
    return BASE_ASCII[j][i]


def wall_cells():
    """所有障碍格（含树）的网格下标 (i, j)"""
    return [(i, j) for j in range(SIZE) for i in range(SIZE) if _cell_char(i, j) != "."]


def wall_points():
    """障碍格的世界坐标，格式与现有 Demo 的 MAZE_POINTS 完全一致：[x, y, 1.5]

    只有它一个真源：QLabs 场景和 VirtualQDrone2 都从这里派生，不可能对不上。
    """
    return [[i + 0.5, j + 0.5, 1.5] for (i, j) in wall_cells()]


def cell_style(i, j):
    """障碍格的外观：'building' 或 'tree'"""
    return "tree" if _cell_char(i, j) == "T" else "building"


def occupancy():
    """占据栅格，与现有 Demo 的 MAZE 约定逐格一致（0=墙, 1=通）

    映射规则与 Demo 的 create_maze 相同：格 (i,j) 的世界坐标是 (i+0.5, j+0.5)，
    对应 MAZE[i+1, j+1]。
    """
    maze = np.ones((SIZE + 2, SIZE + 2))
    for k in range(SIZE + 2):            # 外圈是边界墙
        maze[0, k] = 0
        maze[k, 0] = 0
        maze[SIZE + 1, k] = 0
        maze[k, SIZE + 1] = 0
    for (i, j) in wall_cells():
        maze[i + 1, j + 1] = 0
    return maze


def free_cells():
    return [(i, j) for j in range(SIZE) for i in range(SIZE) if _cell_char(i, j) == "."]


def shortest_path():
    """BFS 求起点到终点的最短步数；返回 (步数, 路径)。无解返回 (None, None)。

    纯 Python，不需要 QLabs —— 用作地图合法性的离线护栏。
    """
    from collections import deque
    start, goal = (0, 0), (SIZE - 1, SIZE - 1)
    if _cell_char(*start) != "." or _cell_char(*goal) != ".":
        return None, None
    q = deque([(start, [start])]); seen = {start}
    while q:
        (i, j), path = q.popleft()
        if (i, j) == goal:
            return len(path) - 1, path
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < SIZE and 0 <= nj < SIZE and _cell_char(ni, nj) == "." and (ni, nj) not in seen:
                seen.add((ni, nj)); q.append(((ni, nj), path + [(ni, nj)]))
    return None, None


def reachable_from_start():
    """从起点可达的所有格（用于检查有没有"够不到的孤岛"）"""
    from collections import deque
    start = (0, 0)
    if _cell_char(*start) != ".":
        return set()
    q = deque([start]); seen = {start}
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < SIZE and 0 <= nj < SIZE and _cell_char(ni, nj) == "." and (ni, nj) not in seen:
                seen.add((ni, nj)); q.append((ni, nj))
    return seen


def load_ascii(ascii_list, verbose=True):
    """校验并装载一份 ASCII 地图（覆盖模块级的 BASE_ASCII）。

    现有所有函数（wall_cells / occupancy / shortest_path / build_scene …）
    都经由 _cell_char 读 BASE_ASCII，所以装载后**自动全部跟随**。

    校验不过会抛 ValueError 并说明原因 —— 学生改坏了能立刻看到问题。
    """
    global BASE_ASCII
    if len(ascii_list) != SIZE:
        raise ValueError(f"地图应有 {SIZE} 行，实际 {len(ascii_list)} 行")
    for j, row in enumerate(ascii_list):
        if len(row) != SIZE:
            raise ValueError(f"第 {j} 行应有 {SIZE} 个字符，实际 {len(row)} 个")
        bad = set(row) - VALID_CHARS
        if bad:
            raise ValueError(f"第 {j} 行含非法字符 {sorted(bad)}，只允许 {sorted(VALID_CHARS)}")
    BASE_ASCII = list(ascii_list)
    if verbose:
        print(f"已装载地图：{SIZE}x{SIZE}，障碍 {len(wall_cells())} 格")
    return BASE_ASCII


def components():
    """所有可通行区域的连通分量，按大小从大到小返回 [set_of_cells, ...]"""
    remaining = set(free_cells())
    comps = []
    while remaining:
        seed = next(iter(remaining))
        remaining.discard(seed)
        comp, stack = set(), [seed]
        while stack:
            i, j = stack.pop()
            comp.add((i, j))
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (i + di, j + dj)
                if nb in remaining:
                    remaining.discard(nb)
                    stack.append(nb)
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def mission_report(deliveries, start=(0, 0), goal=None, verbose=True):
    """体检 + 堵路测试（纯离线，不需要 QLabs）。返回 True/False（是否通过验收）。

    输入变量：
        deliveries: 有序投递点格子列表
        start: 起点格子
        goal: 终点格子；默认 (SIZE-1, SIZE-1)
    """
    from Module import Astar                 # 惰性导入，避免循环依赖

    start = tuple(start)
    goal = (SIZE - 1, SIZE - 1) if goal is None else tuple(goal)
    ok = True

    def say(*a):
        if verbose:
            print(*a)

    say("=" * 62)
    say("城市体检")
    say("=" * 62)

    wc = wall_cells()
    trees = sum(1 for c in wc if cell_style(*c) == "tree")
    say(f"  尺寸        {SIZE} x {SIZE}")
    say(f"  障碍        {len(wc)} 格（楼 {len(wc)-trees}，树 {trees}）")
    say(f"  可通行      {len(free_cells())} 格")

    comps = components()
    say(f"  连通分量    {len(comps)} 个（最大 {len(comps[0])} 格）")
    if len(comps) > 1:
        say(f"    ⚠ 有 {len(comps)-1} 个与主区域**不通**的孤立区，例如 "
            f"{[sorted(c)[0] for c in comps[1:5]]} …")
        ok = False

    de = dead_ends()
    say(f"  死胡同      {len(de)} 个" + (f"  {de[:8]}{' …' if len(de) > 8 else ''}" if de else ""))

    steps = shortest_path()[0]
    say(f"  起点->终点  {'✗ 不可达' if steps is None else str(steps) + ' 步'}")

    say("")
    say("=" * 62)
    say("堵路测试（固定任务必须仍然可完成）")
    say("=" * 62)
    say("  任务：" + " -> ".join(str(p) for p in (start, *deliveries, goal)))

    free = set(free_cells())
    for name, pt in [("起点", start), ("终点", goal)] + \
                    [(f"投递点{k+1}", tuple(d)) for k, d in enumerate(deliveries)]:
        if pt not in free:
            say(f"  ✗ {name} {pt} 落在墙上或越界")
            ok = False

    cells, legs = Astar.plan_route(start, deliveries, goal=goal)
    for k, L in enumerate(legs, 1):
        if L["ok"]:
            say(f"    第{k}段 {L['from']} -> {L['to']}: {L['steps']:>3} 步  ✓")
        else:
            say(f"    第{k}段 {L['from']} -> {L['to']}:  ✗ 被堵死")
    if cells is None:
        say("  ✗ 任务不可解 —— 路线被堵死了")
        ok = False
    else:
        say(f"  任务总长 {len(cells)-1} 步  ✓ 可解")

    say("")
    say(f"  ==> 验收 {'✓ 通过' if ok else '✗ 不通过'}")
    return ok, (cells, legs)


# ---------------------------------------------------------------------------
# 场景构建
# ---------------------------------------------------------------------------

def preflight(qlabs):
    """建图前先确认 QLabs 场景真的可用。

    QLabs 的 spawn 失败时**不抛异常**，只返回非 0 状态码（-1 = 通讯错误）。
    如果世界被卸载（例如客户端被强杀异常断开），后续所有 spawn 都会静默失败，
    而调用方如果只看自己的打印，会误以为"建图成功"。这里显式挡掉这种情况。
    """
    from qvl.basic_shape import QLabsBasicShape
    st = QLabsBasicShape(qlabs).spawn(location=[-60, -60, -60], rotation=[0, 0, 0], scale=[1, 1, 1])
    if isinstance(st, tuple):
        st = st[0]
    if st != 0:
        raise RuntimeError(
            f"QLabs 场景不可用（spawn 返回 {st}；-1 表示通讯错误）。\n"
            "  最常见原因：QLabs 卸载了世界。请在 QLabs 里重新打开 QDrone2 / Plane 空世界。\n"
            "  另一个坑：不要用外部 timeout 强杀连着 QLabs 的 Python 进程 —— 客户端异常\n"
            "  断开会让 QLabs 退回世界选择界面。让它优雅退出（例如用 --seconds）。"
        )


def build_scene(qlabs, city=None, pedestrians="off", verbose=True, decor=None, deliveries=None):
    """把地图搭进 QLabs。返回 City 实例（用于之后 stop_pedestrians）。

    输入变量：
        qlabs: QuanserInteractiveLabs
        city: (可选) 复用一个已有的 City 实例
        pedestrians: 'off' 或 'dynamic'。探针实测行人**永远挡路**，
                     所以没有"不挡路的景观行人"这一档（见模块 docstring）。
        decor: (可选) 装饰列表（见 apply_decor 的 schema）。不给就用默认的 decorate()。
        deliveries: (可选) 有序投递点格子列表，会在对应格子放标记。
    """
    from Module.City import City
    from qvl.person import QLabsPerson
    import time

    preflight(qlabs)                 # 场景不可用就立刻报错，别静默"建完图"
    city = city or City(qlabs)
    city.clean_all()

    # 1) 边界墙。高度特意压到 1.5 m —— 实测 1.0 m 就足以在 z=1.0 挡住无人机，
    #    留 0.5 m 余量；墙不高，场外的"郊区环"才看得见（21 m 会把场外全遮住）。
    BH = 1.5
    city.create_building([-0.5, SIZE / 2, BH / 2], [0, 0, 0], [1, SIZE + 2, BH], [0.1, 0.1, 0.1], roughness=0.01)
    city.create_building([SIZE / 2, -0.5, BH / 2], [0, 0, 0], [SIZE, 1, BH], [0.1, 0.1, 0.1], roughness=0.01)
    city.create_building([SIZE + 0.5, SIZE / 2, BH / 2], [0, 0, 0], [1, SIZE + 2, BH], [0.1, 0.1, 0.1], roughness=0.01)
    city.create_building([SIZE / 2, SIZE + 0.5, BH / 2], [0, 0, 0], [SIZE, 1, BH], [0.1, 0.1, 0.1], roughness=0.01)

    # 2) 障碍格：楼 或 树。楼的高度与颜色按 (i,j) 确定性变化 ——
    #    占地仍是 1x1 格，**碰撞行为逐格不变**，但整座城市不再是一排一样的灰方块。
    for (i, j) in wall_cells():
        x, y = i + 0.5, j + 0.5
        if cell_style(i, j) == "tree":
            city.create_tree([x, y, 0.0], scale=1.0)
        else:
            h, color = building_style(i, j)
            city.create_building([x, y, h / 2], [0, 0, 0], [1, 1, h], color)

    # 3) 起点 / 终点
    city.create_building([0.5, 0.5, 0.05], [0, 0, 0], [1, 1, 0.1], [1, 0, 0])
    city.create_building([SIZE - 0.5, SIZE - 0.5, 0.05], [0, 0, 0], [1, 1, 0.1], [0, 0, 1])

    # 4) 投递点标记：橙色矮地砖 + 参考坐标系。
    #    场景里**没有任何文字标签**，只能靠颜色 + 标记来区分（见模块 docstring）。
    if deliveries:
        for (i, j) in deliveries:
            city.create_building([i + 0.5, j + 0.5, 0.04], [0, 0, 0], [1, 1, 0.08], DELIVERY_COLOR)
            city.create_reference_frame([i + 0.5, j + 0.5, 1.2])

    # 5) 装饰层（只生成 actor，不影响 occupancy）
    if decor is None:
        decorate(qlabs, city)
    else:
        n_dec = apply_decor(city, decor)
        if verbose:
            print(f"  装饰：来自 DECOR 列表（{n_dec} 项）")

    # 6) 场外郊区环 —— 全部落在边界之外，结构上不可能影响可解性
    n_out = outskirts(city)
    if verbose:
        print(f"  郊区：边界外 {n_out} 个组件")

    # 5) 行人
    if pedestrians == "dynamic":
        # 必须"先把所有行人建完，再统一启动线程"。
        # 原因：QLabs 的 Python 客户端**不是线程安全的** —— 行人线程在用 socket 调
        # move_to 的同时，主线程若还在用同一个 socket 建图，会直接段错误（实测，
        # 见 probe/阶段0-结论.md）。所以两段必须严格分开。
        movers = []
        for route in PEDESTRIAN_ROUTES:
            wp = [[p[0], p[1], 0.0] for p in route]
            person = city.create_character(wp[0], [0, 0, 0], [1, 1, 1])
            movers.append((person, wp))
            time.sleep(0.3)
        time.sleep(0.5)                      # 让最后一帧建图指令确认完成
        for person, wp in movers:
            city.periodic_thread(person, wp[0], wp, QLabsPerson.WALK)
        if verbose:
            print(f"  行人：dynamic（{len(PEDESTRIAN_ROUTES)} 条巡逻线，会挡路）")
    elif verbose:
        print("  行人：off")

    return city


def decorate(qlabs, city):
    """装饰层 —— 全部是矮物件（≤0.6），铺在街道上不会挡路

    探针实测：SplineLine、Crosswalk、ReferenceFrame 都不挡路（各 6/6 到达）。
    """
    # 横向街道的中心线与纵向街道的中心线
    for j in range(SIZE):
        if j % 3 == 0:
            city.create_road([[0.5, j + 0.5], [SIZE - 0.5, j + 0.5]], width=0.9)
    for i in range(SIZE):
        if i % 3 == 0:
            city.create_road([[i + 0.5, 0.5], [i + 0.5, SIZE - 0.5]], width=0.9)

    # 几个路口的车道线环
    for (i, j) in [(0, 0), (9, 9), (18, 18), (6, 12)]:
        city.create_lane_line([i + 0.5, j + 0.5], radius=0.35)

    # 几个人行横道
    for (i, j) in [(3, 6), (12, 3), (15, 15)]:
        city.create_crosswalk([i + 0.5, j + 0.5, 0.02])

    # 起终点各放一个参考坐标系（场景里没有文字标签，用它当标记）
    city.create_reference_frame([0.5, 0.5, 1.2])
    city.create_reference_frame([SIZE - 0.5, SIZE - 0.5, 1.2])


# ---- 装饰 schema：刻意只允许"矮件 / 不挡路"的组件 ----
# 楼和树**不在其中** —— 要加就改 ASCII。
# 这样"装饰永不把障碍物摆到走廊上"是**结构上成立**的，不靠约定。
DECOR_DISPATCH = {
    "road":      lambda c, d: c.create_road(d["points"], d.get("width", 0.6),
                                            d.get("color", [0.15, 0.15, 0.15]), d.get("z", 0.02)),
    "lane_line": lambda c, d: c.create_lane_line(d["center"], d["radius"],
                                                 d.get("line_width", 0.08),
                                                 d.get("color", [1.0, 0.85, 0.0]), d.get("z", 0.03)),
    "crosswalk": lambda c, d: c.create_crosswalk(d["location"], d.get("rotation", [0, 0, 0]),
                                                 d.get("scale", [1, 1, 1]), d.get("configuration", 0)),
    "frame":     lambda c, d: c.create_reference_frame(d["location"], d.get("rotation", [0, 0, 0]),
                                                       d.get("scale", [1, 1, 1])),
    "cone":      lambda c, d: c.create_cone(d["location"], d.get("rotation", [0, 0, 0]),
                                            d.get("scale", [1, 1, 1])),
    "stop_sign": lambda c, d: c.create_stop_sign(d["location"], d.get("rotation", [0, 0, 0]),
                                                 d.get("scale", [1, 1, 1])),
    "wall":      lambda c, d: c.create_wall_block(d["location"], d.get("rotation", [0, 0, 0]),
                                                  d.get("scale", [1, 1, 3])),
}


def apply_decor(city, decor):
    """按 DECOR 列表生成装饰，返回成功生成的项数。

    schema 见 DECOR_DISPATCH。不认识的 kind 会被跳过并提示，不会中断整张地图的构建。
    """
    n = 0
    for k, d in enumerate(decor):
        kind = d.get("kind")
        fn = DECOR_DISPATCH.get(kind)
        if fn is None:
            print(f"  装饰第 {k} 项的 kind={kind!r} 不认识，已跳过"
                  f"（可用：{sorted(DECOR_DISPATCH)}）")
            continue
        fn(city, d)
        n += 1
    return n


def outskirts(city, seed=20261012, budget=70):
    """场外郊区环 —— **全部落在 19x19 可飞区域之外**。

    因为这些格子根本不在可飞区域里，**结构上不可能影响 occupancy()、
    可解性或堵路测试**。高度随便加，都碰不到无人机。

    用固定种子的 random.Random，保证每次建图完全可复现（场景只建一次，
    建出来的样子必须稳定）。

    输入变量：
        seed: 随机种子
        budget: 最多放几个组件（每个约 66.7 ms，70 个约 4.7 秒）
    """
    rng = random.Random(seed)
    M = 4                                          # 郊区带宽（格）
    lo, hi = -M, SIZE + M

    # 一圈环路（SplineLine，矮，不挡路）
    city.create_road([[lo + 0.5, lo + 0.5], [hi + 0.5, lo + 0.5],
                      [hi + 0.5, hi + 0.5], [lo + 0.5, hi + 0.5],
                      [lo + 0.5, lo + 0.5]], width=2.0)

    # 郊区带里的候选位置（步长 2，避免太密）
    cand = []
    for i in range(lo, hi + 1, 2):
        for j in range(lo, hi + 1, 2):
            if i < 0 or i >= SIZE or j < 0 or j >= SIZE:   # 只要边界之外的
                cand.append((i, j))
    rng.shuffle(cand)

    n = 0
    for (i, j) in cand[:budget]:
        x, y = i + 0.5, j + 0.5
        if rng.random() < 0.30:
            city.create_tree([x, y, 0.0], scale=rng.uniform(0.9, 1.5))
        else:
            h = rng.choice([1.5, 2.0, 2.5, 3.0, 4.0])
            color = BUILDING_PALETTE[(i * 3 + j * 7) % len(BUILDING_PALETTE)]
            city.create_building([x, y, h / 2], [0, 0, 0], [1.6, 1.6, h], color)
        n += 1

    # 一个象限放几栋高楼，做出"市中心天际线"
    for (i, j, h) in [(21, 21, 9.0), (23, 25, 12.0), (25, 21, 7.0), (22, 27, 10.0)]:
        city.create_building([i + 0.5, j + 0.5, h / 2], [0, 0, 0], [2.2, 2.2, h],
                             BUILDING_PALETTE[(i + j) % len(BUILDING_PALETTE)])
        n += 1
    return n


# ---------------------------------------------------------------------------
# 自检（不需要 QLabs）
# ---------------------------------------------------------------------------

def dead_ends():
    """死胡同：可通行、且只有一个可通行邻居的格子"""
    out = []
    for (i, j) in free_cells():
        n = 0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < SIZE and 0 <= nj < SIZE and _cell_char(ni, nj) == ".":
                n += 1
        if n == 1:
            out.append((i, j))
    return out


def report():
    steps, path = shortest_path()
    reach = reachable_from_start()
    free = set(free_cells())
    isolated = free - reach
    de = dead_ends()
    print(f"地图自检（纯离线）")
    print(f"  障碍格 {len(wall_cells())} / {SIZE*SIZE}   （其中树 {sum(1 for c in wall_cells() if cell_style(*c)=='tree')} 个）")
    print(f"  可通行 {len(free)} 格，从起点可达 {len(reach)} 格")
    print(f"  最短路径 {steps} 步（曼哈顿下界 {2*(SIZE-1)}）")
    print(f"  够不到的孤岛: {len(isolated)} 格 {'✓ 无' if not isolated else sorted(isolated)[:10]}")
    print(f"  死胡同 {len(de)} 个")
    return steps


if __name__ == "__main__":
    report()
