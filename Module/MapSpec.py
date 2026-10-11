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

import numpy as np

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


# ---------------------------------------------------------------------------
# 场景构建
# ---------------------------------------------------------------------------

def build_scene(qlabs, city=None, pedestrians="off", verbose=True):
    """把地图搭进 QLabs。返回 City 实例（用于之后 stop_pedestrians）。

    输入变量：
        qlabs: QuanserInteractiveLabs
        city: (可选) 复用一个已有的 City 实例
        pedestrians: 'off' 或 'dynamic'。探针实测行人**永远挡路**，
                     所以没有"不挡路的景观行人"这一档（见模块 docstring）。
    """
    from Module.City import City
    from qvl.person import QLabsPerson
    import time

    city = city or City(qlabs)
    city.clean_all()

    # 1) 边界（与现有 Demo 的 build_city 完全相同的做法）
    city.create_building([-0.5, SIZE / 2, 10.5], [0, 0, 0], [1, SIZE + 2, 21], [0.1, 0.1, 0.1], roughness=0.01)
    city.create_building([SIZE / 2, -0.5, 10.5], [0, 0, 0], [SIZE, 1, 21], [0.1, 0.1, 0.1], roughness=0.01)
    city.create_building([SIZE + 0.5, SIZE / 2, 10.5], [0, 0, 0], [1, SIZE + 2, 21], [0.1, 0.1, 0.1], roughness=0.01)
    city.create_building([SIZE / 2, SIZE + 0.5, 10.5], [0, 0, 0], [SIZE, 1, 21], [0.1, 0.1, 0.1], roughness=0.01)

    # 2) 障碍格：楼 或 树（碰撞行为相同，外观不同）
    for (i, j) in wall_cells():
        x, y = i + 0.5, j + 0.5
        if cell_style(i, j) == "tree":
            city.create_tree([x, y, 0.0], scale=1.0)
        else:
            city.create_building([x, y, 1.5], [0, 0, 0], [1, 1, 3], [0.3, 0.3, 0.3])

    # 3) 起点 / 终点
    city.create_building([0.5, 0.5, 0.05], [0, 0, 0], [1, 1, 0.1], [1, 0, 0])
    city.create_building([SIZE - 0.5, SIZE - 0.5, 0.05], [0, 0, 0], [1, 1, 0.1], [0, 0, 1])

    # 4) 装饰层（只生成 actor，不影响 occupancy）
    decorate(qlabs, city)

    # 5) 行人
    if pedestrians == "dynamic":
        for route in PEDESTRIAN_ROUTES:
            wp = [[p[0], p[1], 0.0] for p in route]
            person = city.create_character(wp[0], [0, 0, 0], [1, 1, 1])
            time.sleep(0.3)
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
