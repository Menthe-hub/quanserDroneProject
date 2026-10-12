"""地图定义与编译器（阶段 2）

与现有 Demo 完全解耦：不 import QLearningSimulation，也不改动它。
现有两个 Demo 一行未变，本模块是新增的独立地图来源。

## 三层结构

    走廊格 / 障碍格（ASCII）──► compile() ──► wall_points() / occupancy()
                                          └─► build_scene()   生成 QLabs 场景
    装饰层 ─────────────────────────► decorate()   只生成 actor，**永不写入 occupancy**

不变量：`occupancy()` 只由 ASCII 决定，装饰层没有任何路径能影响它。
所以"加装饰不破坏算法"是结构上成立的，不是靠约定。

## 实测得出的约束

巡航高度取 **z=8.5**，高于所有楼（4~7.8 m）。由此：

* 送货点**可以摆在楼顶** —— 无人机飞到它正上方即可
* **障碍不构成约束，两点之间直线可达** —— 寻路没有难度，实验重心是**调度**

其余：

* 楼高必须 **≥4.0 m**（`HEIGHT_JUMPS` 给 4.0~7.8），矮于巡航高度的会被飞越
* 行人（约 1.8 m）在巡航高度之下，**不挡路**。行人本身"永远挡路、关不掉"，
  只能靠"飞在它之上"绕开
* 矮装饰（路面 / 车道线 / 人行横道，z ≤ 0.04）可随便铺，不挡路
* `QLabsWalls`（泡沫板墙）无法做成障碍，是纯视觉组件
* 建图 **66.7 ms/块**，只能建一次，绝不能放进 `reset_environment`
"""

import math
import random
import threading

import numpy as np

# 把 QLabs 的访问串行化（幂等）。只在真正连 QLabs 时才起作用，虚拟模式无影响。
# 不加的话：用了行人线程（dynamic 模式）后，主线程的请求会被线程抢走应答，
# 表现为无人机误报碰撞 / spawn 静默失败。详见 Module/QLabsLock.py。
from Module import QLabsLock  # noqa: F401

# ---------------------------------------------------------------------------
# 城市几何：**一个**十字路口 + 四个角落街区
# ---------------------------------------------------------------------------
# 剖面（街道是有厚度的，人行道是真空格，楼退到街里面）：
#
#     |← 人行道 2 →|← 车行道 4 →|← 人行道 2 →|      走廊总宽 8 格，居中
#
# 四条路从正中的路口向四个角放射出去，**只有中间这一个路口**。
# 四个 16x16 街角里，一个是广场（`PLAZA_ORIGIN`），另外三个盖楼。
SW = 2                            # 人行道宽度（每侧，格）
CARRIAGE = 4                      # 车行道宽度（格，双向四车道）
CORRIDOR = 2 * SW + CARRIAGE      # 街道走廊总宽 = 8
SIZE = 40                         # 地图边长
CENTER = SIZE // 2                # 路口中心 = 20

CAR_LO, CAR_HI = CENTER - CARRIAGE // 2, CENTER + CARRIAGE // 2   # 车行道 [18, 22)
COR_LO, COR_HI = CENTER - CORRIDOR // 2, CENTER + CORRIDOR // 2   # 走廊  [16, 24)
BLOCK = COR_LO                                                    # 每个街角 16x16
PLAZA_ORIGIN = (COR_HI, 0)        # 广场占东北那个街角（路口旁边）
PLAZA_SIZE = BLOCK

# 起点 / 终点放在**路上**（四个街角一直盖到图边缘，角上未必是空地）。
# 语义也更自然：无人机从街的这头起飞，飞到街的那头。
START_CELL = (CENTER - 1, 0)              # 纵向路北端
GOAL_CELL = (SIZE - 1, CENTER - 1)        # 横向路东端

ORIGINE = [START_CELL[0] + 0.5, START_CELL[1] + 0.5]
GOAL = [GOAL_CELL[0] + 0.5, GOAL_CELL[1] + 0.5]

# 巡航高度，取在所有楼之上（楼 4.0~7.8）。见文件头"约束"一节 ——
# 这是"送货点可放楼顶 + 寻路无难度 + 学生写调度"这套设计的支点。
CRUISE_Z = 8.5

CITY_SEED = 20261012              # 固定种子 -> 每次建图完全可复现
BLOCK_FILL = 0.85                 # 街区内每个楼位的落成概率。不设 1.0：
                                  # 全盖满时换种子只改变楼的尺寸、布局完全一样

# 楼高按**楼栋编号**轮转 —— 编号相邻 = 位置相邻，所以挨着的两栋必然差一档，
# 一眼就能看出来是两栋楼，而不是一片高度雷同的大板。
# 最低 4.0 m：巡航高度是 3.0，低于 4.0 的楼会被飞越（碰撞层就和 occupancy() 对不上了）。
HEIGHT_JUMPS = [0.0, 2.6, 1.0, 3.8, 1.6, 3.0]      # -> 4.0 6.6 5.0 7.8 5.6 7.0

# 批量建楼时**不等待应答**：每个 spawn 的确认要 ~66.7 ms，一座城就是几十秒；
# 不等则单次开销降到 ~0.02 ms。代价：拿不到状态码（`preflight()` 已挡掉
# "世界没加载"这种最常见的失败），且应答会堆在接收缓冲里，必须定期
# flush_receive()，否则冲爆后读回全是垃圾。
FAST_BUILD = True
FLUSH_EVERY = 40                                   # 每多少个组件清一次接收缓冲
BULK_ACTOR_BASE = 3000                             # 批量方块自用的 actorNumber 起点
                                                   # （取大值，避开 QLabs 自动分配的编号）


def _bulk_box(qlabs, n, location, scale, color, roughness=0.8):
    """批量用的方块：**自己指定 actorNumber + 不等确认**。

    ⚠ 必须用 `spawn_id` 而不是 `spawn`。`QLabsActor.spawn()` 在
    `waitForConfirmation=False` 时**根本不会设置 `self.actorNumber`**
    （源码里那条分支直接 `return 0, -1`），于是紧接着的
    `set_material_properties` 打到了别的 actor 上 —— **颜色全部不生效**，
    整座城都是默认灰。`spawn_id` 即使不等确认也会把编号记下来。

"""
    from qvl.basic_shape import QLabsBasicShape
    b = QLabsBasicShape(qlabs)
    b.spawn_id(actorNumber=BULK_ACTOR_BASE + n, location=location, rotation=[0, 0, 0],
               scale=scale, configuration=QLabsBasicShape.SHAPE_CUBE,
               waitForConfirmation=False)
    b.set_material_properties(color=color, roughness=roughness, waitForConfirmation=False)
    return b
PLAZA_WALKERS = 5                 # 广场里随机漫步的行人数量


def is_road(i, j):
    """该格是不是车行道（纵向路或横向路，取并集）"""
    return CAR_LO <= i < CAR_HI or CAR_LO <= j < CAR_HI


def is_sidewalk(i, j):
    """该格是不是人行道 —— 紧贴车行道外侧的两条带。

    **人行道是真空格**（无人机也能飞，只是地面有行人），
    它的作用是让楼**退到街里面**，街道才有厚度。
    """
    if is_road(i, j):
        return False
    return (COR_LO <= i < CAR_LO or CAR_HI <= i < COR_HI) or \
           (COR_LO <= j < CAR_LO or CAR_HI <= j < COR_HI)


def block_origins():
    """四个街角的左上角格坐标"""
    return [(i0, j0) for j0 in (0, COR_HI) for i0 in (0, COR_HI)]


def _build_block(rng, grid, i0, j0, fill, owner, ctr):
    """在一个街角里盖楼：**一栋一栋互相分开的大楼** + 巷子。

    楼位格点间距 5、楼最大 4 格 —— 所以每两栋楼之间**至少空 1 格**，
    紧挨着的两栋不会粘成一整块，屋顶各自独立，高度也能差开。

    `owner` 记下每格属于哪一栋，供 `building_units()` 一栋一个盒子地渲染。
    """
    step = 5
    for a in range(0, BLOCK, step):
        for b in range(0, BLOCK, step):
            if BLOCK - a < 3 or BLOCK - b < 3 or rng.random() >= fill:
                continue
            w = rng.randint(3, min(4, BLOCK - a))
            h = rng.randint(3, min(4, BLOCK - b))
            ctr[0] += 1
            for di in range(w):
                for dj in range(h):
                    grid[j0 + b + dj][i0 + a + di] = "#"
                    owner[(i0 + a + di, j0 + b + dj)] = ctr[0]

    # 贯通小巷：随机切 1~3 条，把整块街区再打散一次，巷子成网。
    # 条数随机而不是固定 2 条 —— 否则每个街区的骨架都一样，换种子看不出差别。
    for _ in range(rng.randint(1, 3)):
        if rng.random() < 0.5:
            a = rng.randrange(1, BLOCK - 1)
            for b in range(BLOCK):
                owner.pop((i0 + a, j0 + b), None)
                grid[j0 + b][i0 + a] = "."
        else:
            b = rng.randrange(1, BLOCK - 1)
            for a in range(BLOCK):
                owner.pop((i0 + a, j0 + b), None)
                grid[j0 + b][i0 + a] = "."

    # 街角里零星种几棵树
    for _ in range(rng.randint(0, 2)):
        a, b = rng.randrange(BLOCK), rng.randrange(BLOCK)
        if grid[j0 + b][i0 + a] == ".":
            grid[j0 + b][i0 + a] = "T"


def _plant_plaza(rng, grid, i0, j0):
    """广场：中间整片开敞，树**零散**点在四周（不排成整齐的一圈）。

    在外围环带随机撒点，任意两棵至少隔 3 格；中间留空当活动区。
    """
    for a in range(BLOCK):
        for b in range(BLOCK):
            grid[j0 + b][i0 + a] = "."

    lo, hi = 2, BLOCK - 3          # 允许种树的坐标范围
    inner_lo, inner_hi = 5, BLOCK - 6
    pts = []
    for _ in range(300):
        if len(pts) >= 22:
            break
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        if inner_lo <= a <= inner_hi and inner_lo <= b <= inner_hi:
            continue                                   # 中间留空
        if all(abs(a - x) + abs(b - y) >= 3 for x, y in pts):   # 最小间距
            pts.append((a, b))
    for a, b in pts:
        grid[j0 + b][i0 + a] = "T"


def generate_city(seed=CITY_SEED, fill=BLOCK_FILL):
    """生成一座城市，返回 ASCII 列表（可直接喂给 load_ascii）。

    固定种子 -> 完全可复现（场景只建一次，建出来的样子必须稳定）。
    三个街角盖楼（成片的楼块 + 巷子 + 贯通小巷），东北角是广场。

    输入变量：
        seed: 随机种子
        fill: 每个楼位的落成概率（0~1）。剩下的留成空地，
              **空地是无人机可以穿行的**，所以巷子也参与寻路
    """
    # 注意：本函数在 import 时就会被调用，那时 invalidate_cache 还没定义，
    # 所以这里直接操作全局量，不能调函数。
    global _CELL_OWNER, _GENERATED, _block_cache
    rng = random.Random(seed)
    grid = [["." for _ in range(SIZE)] for _ in range(SIZE)]
    owner, ctr = {}, [0]

    for (i0, j0) in block_origins():
        if (i0, j0) == PLAZA_ORIGIN:
            _plant_plaza(rng, grid, i0, j0)
        else:
            _build_block(rng, grid, i0, j0, fill, owner, ctr)

    # 连通性修复：从道路洪泛，够不到的空地一律当楼填掉。
    # 否则会留下"四面被楼围死"的天井 —— 那些格在 occupancy() 里算可通行，
    # 但 A* 永远到不了，体检报告会报孤岛，虚拟/真机也会对不上。
    from collections import deque
    seen = set()
    q = deque()
    for j in range(SIZE):
        for i in range(SIZE):
            if is_road(i, j):
                seen.add((i, j)); q.append((i, j))
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if (0 <= ni < SIZE and 0 <= nj < SIZE
                    and (ni, nj) not in seen and grid[nj][ni] == "."):
                seen.add((ni, nj)); q.append((ni, nj))
    for j in range(SIZE):
        for i in range(SIZE):
            if grid[j][i] == "." and (i, j) not in seen:
                grid[j][i] = "#"

    rows = ["".join(row) for row in grid]
    _CELL_OWNER = owner
    _GENERATED = rows
    _block_cache = None
    return rows


# 地图本体：由路网规则生成。load_ascii() 可以整体覆盖它（学生改图走这条路）。
BASE_ASCII = generate_city()

assert len(BASE_ASCII) == SIZE, f"ASCII 行数应为 {SIZE}"
for _r, _row in enumerate(BASE_ASCII):
    assert len(_row) == SIZE, f"第 {_r} 行长度 {len(_row)}，应为 {SIZE}"

# ---- 外观与合法性常量 ----

VALID_CHARS = {".", "#", "T"}          # 学生能用的字符集

# 楼的外墙配色。压过饱和度的建材色，有区分度但不花哨。
BUILDING_PALETTE = [
    [0.62, 0.61, 0.58],   # 浅水泥
    [0.51, 0.49, 0.45],   # 石材
    [0.45, 0.33, 0.28],   # 红砖
    [0.35, 0.40, 0.47],   # 青灰（玻璃幕墙）
    [0.29, 0.29, 0.31],   # 深灰
    [0.60, 0.48, 0.36],   # 砂色
    [0.40, 0.46, 0.42],   # 灰绿
    [0.69, 0.67, 0.63],   # 米白
    [0.46, 0.36, 0.34],   # 暗砖
    [0.33, 0.36, 0.39],   # 石板
]
DELIVERY_COLOR = [1.0, 0.55, 0.0]      # 送货点地砖：橙色（场景里没有文字标签，只能靠颜色）
PICKUP_COLOR = [0.10, 0.85, 1.0]       # 取货点地砖：青色，与送货点区分开


_block_cache = None
_CELL_OWNER = {}      # {(i,j): 楼栋号}，由 generate_city 填；手工图则留空
_GENERATED = None     # generate_city 最近产出的那份 ASCII


def invalidate_cache():
    """地图变了（load_ascii / generate_city）之后必须清一次楼块缓存"""
    global _block_cache
    _block_cache = None


def _components(cells):
    """把一堆格子按 4 邻域切成连通块，返回 [[cell, ...], ...]"""
    seen, out = set(), []
    for seed in cells:
        if seed in seen:
            continue
        comp, stack = [], [seed]
        seen.add(seed)
        while stack:
            ci, cj = stack.pop()
            comp.append((ci, cj))
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (ci + di, cj + dj)
                if nb in cells and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        out.append(comp)
    return out


def building_units():
    """一栋一栋的楼：返回 [(cells, 高度, 颜色), ...]，cells 是这栋占的格子列表。

    **同一栋楼里每个格子的高度和颜色完全一样** —— 这是"屋顶平整"的关键。
    以前按 (i,j) 逐格算高度，一栋 3x3 的楼九个格子九个高度，屋顶像一排锯齿。

    **楼栋的切法**：优先用 `generate_city()` 放矩形时记下的编号（编号相邻 = 位置相邻，
    所以挨着的两栋必然差一档高度）；手工地图没有编号，就退回按连通块切。
    """
    cells = {(i, j) for (i, j) in wall_cells() if cell_style(i, j) != "tree"}
    if _CELL_OWNER and cells <= set(_CELL_OWNER):
        groups = []
        by_id = {}
        for c in cells:
            by_id.setdefault(_CELL_OWNER[c], []).append(c)
        for bid, comp in sorted(by_id.items()):
            h = 4.0 + HEIGHT_JUMPS[bid % len(HEIGHT_JUMPS)]
            color = BUILDING_PALETTE[bid % len(BUILDING_PALETTE)]
            groups.append((comp, h, color))
    else:
        for k, comp in enumerate(_components(cells)):
            mi = int(sum(c[0] for c in comp) / len(comp))
            mj = int(sum(c[1] for c in comp) / len(comp))
            h = 4.0 + HEIGHT_JUMPS[(mi * 31 + mj * 17) % len(HEIGHT_JUMPS)]
            color = BUILDING_PALETTE[(mi * 5 + mj * 11) % len(BUILDING_PALETTE)]
            groups.append((comp, h, color))
    return groups


def building_blocks():
    """{(i, j): (高度, 颜色)} —— `building_units()` 的按格展开版"""
    global _block_cache
    if _block_cache is None:
        out = {}
        for comp, h, color in building_units():
            for c in comp:
                out[c] = (h, color)
        _block_cache = out
    return _block_cache


def building_style(i, j):
    """这一格所属那栋楼的高度与颜色（同一栋楼里逐格相同 -> 屋顶平整）"""
    return building_blocks().get((i, j), (5.0, BUILDING_PALETTE[0]))


def roof_top(i, j):
    """这一格上方最高物体的顶面高度（空地是 0）。

    送货点可以摆在楼顶上，这时地砖要画在**楼顶**而不是埋在地面里。
    """
    ch = _cell_char(i, j)
    if ch == "#":
        return building_style(i, j)[0]
    if ch == "T":
        return 3.0                    # 树 ~3 m，标记压在上面免得埋进树冠
    return 0.0


def intersections():
    """**正中间那一个**十字路口，返回 [(c, d)]（c/d 是走廊原点）。

    路口本体是 x ∈ [CAR_LO, CAR_HI) × y ∈ [CAR_LO, CAR_HI) 的 4x4 方块。
    """
    return [(COR_LO, COR_LO)]


def crosswalk_spots(c, d):
    """一个路口的 4 条斑马线：(世界坐标, 绕 z 的旋转, 需要拉伸的倍数)。

    每条都跨满车行道宽度（CARRIAGE 格）—— 斑马线对象 scale=1 时渲染宽 2 格，
    所以拉伸 CARRIAGE/2 倍。放在路口**四个进场方向**，贴着路口边。
    """
    k = CARRIAGE / 2.0
    cx = c + SW + CARRIAGE / 2.0          # 车行道中心线
    cy = d + SW + CARRIAGE / 2.0
    south = d + SW / 2.0                  # 路口以南的人行道上
    north = d + SW + CARRIAGE + SW / 2.0
    west = c + SW / 2.0
    east = c + SW + CARRIAGE + SW / 2.0
    return [
        ([cx, south, 0.02], [0, 0, 0], k),                  # 南进场：横跨南北向车道
        ([cx, north, 0.02], [0, 0, 0], k),                  # 北进场
        ([west, cy, 0.02], [0, 0, math.pi / 2], k),         # 西进场：横跨东西向车道
        ([east, cy, 0.02], [0, 0, math.pi / 2], k),         # 东进场
    ]


def sidewalk_lines():
    """人行道的中心线坐标。返回 (纵向的 x 列表, 横向的 y 列表)。

    两条带 [COR_LO, CAR_LO) 和 [CAR_HI, COR_HI) 的中心各一条。
    """
    return [COR_LO + SW / 2.0, COR_HI - SW / 2.0], [COR_LO + SW / 2.0, COR_HI - SW / 2.0]


def pedestrian_routes():
    """**沿人行道走**的行人路线（世界坐标 [[x,y], ...]）。

    因为无人机巡航在 z=3.0、行人约 1.8 m，**行人走到哪里都不会挡路**
    （实测 0/2 碰撞），所以路线可以自由选。
    """
    xs, ys = sidewalk_lines()
    routes = []
    for x in xs:
        routes.append([[x, 0.5], [x, SIZE - 0.5]])
    for y in ys:
        routes.append([[0.5, y], [SIZE - 0.5, y]])
    return routes


def plaza_walk_points():
    """广场里的随机漫步落脚点（给 `City.random_thread` 用）。"""
    pi, pj = PLAZA_ORIGIN
    pts = []
    for a in (3, 7, 11, 15):
        for b in (3, 7, 11, 15):
            pts.append([pi + a + 0.5, pj + b + 0.5])
    return pts


def crossing_journeys():
    """**过马路的长路线**：沿一条人行道走过来 -> 在路口横穿到对面 -> 沿对面人行道走远。

    约 45 m、走 40 秒上下；起终点都在图的边缘，行人不会在视野正中央出现或消失。
    四条 = 两个方向 x 两条路。
    """
    (c, d), = intersections()
    x_lo, x_hi = sidewalk_lines()[0]              # 两条纵向人行道
    y_lo, y_hi = sidewalk_lines()[1]              # 两条横向人行道
    # 跨纵向路的斑马线在 y = 南/北；跨横向路的斑马线在 x = 西/东
    y_s, y_n = d + SW / 2.0, d + SW + CARRIAGE + SW / 2.0
    x_w, x_e = c + SW / 2.0, c + SW + CARRIAGE + SW / 2.0
    a, b = 0.5, SIZE - 0.5                        # 图的两端
    return [
        # 沿西人行道由北向南 -> 走南斑马线横穿纵向路 -> 沿东人行道继续向南
        [[x_lo, a], [x_lo, y_s], [x_hi, y_s], [x_hi, b]],
        # 由南向北，走北斑马线
        [[x_hi, b], [x_hi, y_n], [x_lo, y_n], [x_lo, a]],
        # 沿北侧人行道由西向东 -> 走东斑马线横穿横向路 -> 继续向东
        [[a, y_lo], [x_w, y_lo], [x_w, y_hi], [b, y_hi]],
        # 由东向西，走西斑马线
        [[b, y_hi], [x_e, y_hi], [x_e, y_lo], [a, y_lo]],
    ]


PEDESTRIAN_ROUTES = pedestrian_routes()
CROSSING_ROUTES = crossing_journeys()


class PedestrianFleet:
    """**会换班的行人**。

    和 `City.periodic_thread`（同一批人永远来回走）不同：这里每个人**走完一趟
    就退休** —— 销毁掉，随即在另一条路线上生成一个新人。街上的人流量是流动的，
    不会出现同一张脸永远在同一条线上来回。

    三种角色：
        "street"  沿人行道走一趟（单程）
        "cross"   沿斑马线过一趟马路
        "plaza"   在广场里随机走 2~4 段

    ⚠ 线程一旦跑起来，主线程就不能再碰同一个 qlabs 连接（QLabs 客户端
    不是线程安全的，见 probe/阶段0-结论.md）。所以 `start()` 必须在**整个场景
    建完之后**才调用。
    """

    def __init__(self, city, n_street=6, n_cross=4, n_plaza=4, seed=None):
        self.city = city
        # seed 默认取**当前**的 CITY_SEED（不能在参数默认值里写死 —— 那是 def
        # 时求值的，换种子后行人路线还是老的）
        self.rng = random.Random((CITY_SEED if seed is None else seed) + 101)
        self.counts = {"street": n_street, "cross": n_cross, "plaza": n_plaza}

    def start(self):
        """为每种角色开固定数量的"工位"，每个工位循环地：生成 -> 走完 -> 销毁"""
        for kind, n in self.counts.items():
            for _ in range(n):
                stop = threading.Event()
                t = threading.Thread(target=self._worker, args=(kind, stop),
                                     daemon=True, name=f"ped-{kind}")
                self.city.register_mover(t, stop)
                t.start()

    def _pick_route(self, kind):
        if kind == "street":
            r = list(self.rng.choice(PEDESTRIAN_ROUTES))
            return r if self.rng.random() < 0.5 else r[::-1]   # 两个方向随机
        if kind == "cross":
            return self.rng.choice(CROSSING_ROUTES)
        # 广场：走够 8~12 段（约 40 秒），别两步就换人
        pts = plaza_walk_points()
        return [self.rng.choice(pts) for _ in range(self.rng.randint(8, 12))]

    def _worker(self, kind, stop):
        from qvl.person import QLabsPerson
        speed = QLabsPerson.WALK
        while not stop.is_set():
            person = None
            try:
                route = self._pick_route(kind)
                person = self.city.create_character(
                    [route[0][0], route[0][1], 0.0], [0, 0, 0], [1, 1, 1])
                prev = route[0]
                for p in route[1:]:
                    if stop.is_set():
                        break
                    person.move_to([p[0], p[1], 0.0], speed)
                    if stop.wait(math.hypot(p[0] - prev[0], p[1] - prev[1]) / speed + 1.0):
                        break
                    prev = p
            except Exception as e:
                print(f"  (行人线程出错: {type(e).__name__}: {e})")
            finally:
                if person is not None:
                    try:
                        person.destroy()
                    except Exception:
                        pass
            stop.wait(self.rng.uniform(0.2, 1.2))     # 错开重建，别让所有人同时换班


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


def nearest_free(cell, radius=8):
    """离 cell 最近的可用格（一圈圈往外找）。找不到返回 None。

    给"自动配点"用：城市是随机生成的，写死的坐标换个种子就会落在楼上。
    """
    i0, j0 = cell
    if _cell_char(i0, j0) == ".":
        return (i0, j0)
    for r in range(1, radius + 1):
        for di in range(-r, r + 1):
            for dj in (-r, r):
                for (i, j) in ((i0 + di, j0 + dj), (i0 + dj, j0 + di)):
                    if 0 <= i < SIZE and 0 <= j < SIZE and _cell_char(i, j) == ".":
                        return (i, j)
    return None


def auto_deliveries():
    """给**当前**这张图自动挑一组合适的送货点：每个盖了楼的街角一个，尽量靠街区中心。

    用在 `--seed` 随机生成城市的时候 —— 那种情况下学生文件里写死的坐标必然对不上。
    """
    out = []
    for (i0, j0) in block_origins():
        if (i0, j0) == PLAZA_ORIGIN:
            continue
        c = nearest_free((i0 + BLOCK // 2, j0 + BLOCK // 2))
        if c is not None:
            out.append(c)
    return out


def shortest_path():
    """BFS 求起点到终点的最短步数；返回 (步数, 路径)。无解返回 (None, None)。

    纯 Python，不需要 QLabs —— 用作地图合法性的离线护栏。
    """
    from collections import deque
    start, goal = START_CELL, GOAL_CELL
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
    start = START_CELL
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
    global BASE_ASCII, _CELL_OWNER
    if len(ascii_list) != SIZE:
        raise ValueError(f"地图应有 {SIZE} 行，实际 {len(ascii_list)} 行")
    for j, row in enumerate(ascii_list):
        if len(row) != SIZE:
            raise ValueError(f"第 {j} 行应有 {SIZE} 个字符，实际 {len(row)} 个")
        bad = set(row) - VALID_CHARS
        if bad:
            raise ValueError(f"第 {j} 行含非法字符 {sorted(bad)}，只允许 {sorted(VALID_CHARS)}")
    BASE_ASCII = list(ascii_list)
    if list(ascii_list) != _GENERATED:      # 学生自己改的图 -> 楼栋编号作废，退回连通块
        _CELL_OWNER = {}
    invalidate_cache()                      # 地图变了，楼块缓存必须作废
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


def mission_report(deliveries, start=START_CELL, goal=None, pickup=None,
                   verbose=True, flying_over=True):
    """体检 + 任务检查（纯离线）。返回 (是否通过, (cells, legs))。

    输入变量：
        deliveries: 有序投递点格子列表
        start: 起点格子
        goal: 终点格子；默认 (SIZE-1, SIZE-1)
        pickup: (可选) 取货点格子。给了就插在送货点**之前** —— 先取后送
        flying_over: 无人机飞在楼之上（当前设计就是），送货点可以在楼顶，
            不存在"堵路"，检查项是各段直线距离。设 False 则退回逐段 A\\* 的
            堵路测试 —— 只有在把巡航高度降到楼之间时才用得上。
    """
    from Module import Astar                 # 惰性导入，避免循环依赖

    start = tuple(start)
    goal = GOAL_CELL if goal is None else tuple(goal)
    pickup = None if pickup is None else tuple(pickup)
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

    chain = (start, *([pickup] if pickup else []), *deliveries, goal)

    if flying_over:
        # 无人机飞在楼之上：障碍不构成约束，两点之间直线可达。
        # 于是"堵路测试"没有意义了 —— 改成报告任务点的位置和直线距离。
        say("")
        say("=" * 62)
        say(f"任务（无人机巡航 z={CRUISE_Z}，在楼顶之上，直线可达）")
        say("=" * 62)
        say(f"  {'  ->  '.join(str(p) for p in chain)}")
        total = 0.0
        named = [("起点", start)]
        if pickup:
            named.append(("取货点", pickup))
        named += [(f"送货点{k+1}", tuple(d)) for k, d in enumerate(deliveries)]
        named.append(("终点", goal))
        for k, (name, pt) in enumerate(named):
            i, j = pt
            if not (0 <= i < SIZE and 0 <= j < SIZE):
                say(f"  ✗ {name} {pt} 越界（图是 {SIZE}x{SIZE}）")
                ok = False
                continue
            ch = _cell_char(i, j)
            where = {".": "空地", "#": "楼顶", "T": "树上"}[ch]
            seg = 0.0
            if k > 0:
                pi, pj = named[k - 1][1]
                seg = math.hypot(i - pi, j - pj)
                total += seg
            say(f"  {name:<8}{pt}  [{where}]  " + (f"上一段 {seg:.1f} m" if k else ""))
        say(f"  直线总里程 {total:.1f} m")
        say("")
        say(f"  ==> {'✓ 任务点都在图内' if ok else '✗ 有任务点越界'}")
        return ok, (None, None)

    say("")
    say("=" * 62)
    say("堵路测试（固定任务必须仍然可完成）")
    say("=" * 62)
    say("  任务：" + " -> ".join(str(p) for p in chain))

    free = set(free_cells())
    named = [("起点", start), ("终点", goal)]
    if pickup:
        named.append(("取货点", pickup))
    named += [(f"投递点{k+1}", tuple(d)) for k, d in enumerate(deliveries)]
    for name, pt in named:
        if pt not in free:
            say(f"  ✗ {name} {pt} 落在墙上或越界")
            ok = False
        elif name == "取货点" and not is_road(*pt):
            say(f"  ⚠ 取货点 {pt} 不在车行道上（建议放在路口）")

    waypoints = ([pickup] if pickup else []) + list(deliveries)
    cells, legs = Astar.plan_route(start, waypoints, goal=goal)
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


def build_scene(qlabs, city=None, pedestrians="off", verbose=True, decor=None,
                deliveries=None, pickup=None):
    """把地图搭进 QLabs。返回 City 实例（用于之后 stop_pedestrians）。

    输入变量：
        qlabs: QuanserInteractiveLabs
        city: (可选) 复用一个已有的 City 实例
        pedestrians: 'off' 或 'dynamic'。行人本身**永远挡路**，但无人机巡航在
                     z=3.0、行人约 1.8 m，**实测行人不挡路**（见模块 docstring），
                     所以 dynamic 现在是安全的。
        decor: (可选) 装饰列表（见 apply_decor 的 schema）。不给就用默认的 decorate()。
        deliveries: (可选) 有序投递点格子列表，会在对应格子放标记。
        pickup: (可选) 取货点格子，用**另一种颜色**标记，与送货点区分。
    """
    from Module.City import City
    from qvl.person import QLabsPerson
    import time

    preflight(qlabs)                 # 场景不可用就立刻报错，别静默"建完图"
    city = city or City(qlabs)
    city.clean_all()

    # 1) **不建边界墙** —— 场景是开放的，四周直接看到郊外。
    #    航迹由规划器保证留在格子里，不需要物理墙来兜底。
    #    （占据栅格 `occupancy()` 的外圈 padding 仍然在，那是给寻路用的，不是可见的墙。）

    # 2) 障碍：树 + 楼。
    #    **一栋楼只 spawn 一个盒子**，不是一格一个 —— 逐格 spawn 是建图慢的主因
    #    （每个 spawn 66.7 ms，424 格就是 28 秒；一栋一个盒子只要 40 来个）。
    #
    #    ⚠ 只有当这栋楼占的格子**正好是一个实心矩形**时才能用一个盒子 ——
    #    否则盒子会盖住本来可以飞的格子，而 occupancy() 不知道，
    #    A* 会规划穿楼路径、无人机必撞。L 形的楼退回逐格。
    n_box = n_cell = n_tree = n_spawn = 0

    def box(loc, scale, color):
        """按 FAST_BUILD 决定走快路还是走 City（慢但会检查返回码）"""
        nonlocal n_spawn
        if FAST_BUILD:
            _bulk_box(qlabs, n_spawn, loc, scale, color)
            n_spawn += 1
            if n_spawn % FLUSH_EVERY == 0:
                qlabs.flush_receive()          # 别让应答堆爆接收缓冲
        else:
            city.create_building(loc, [0, 0, 0], scale, color)

    for comp, h, color in building_units():
        is_, js_ = [c[0] for c in comp], [c[1] for c in comp]
        i0, i1, j0, j1 = min(is_), max(is_), min(js_), max(js_)
        solid = len(comp) == (i1 - i0 + 1) * (j1 - j0 + 1)
        if solid:
            box([(i0 + i1 + 1) / 2, (j0 + j1 + 1) / 2, h / 2],
                [i1 - i0 + 1, j1 - j0 + 1, h], color)
            n_box += 1
        else:
            for (i, j) in comp:
                box([i + 0.5, j + 0.5, h / 2], [1, 1, h], color)
                n_cell += 1
    for (i, j) in wall_cells():
        if cell_style(i, j) == "tree":
            city.create_tree([i + 0.5, j + 0.5, 0.0], scale=1.0)
            n_tree += 1
    if FAST_BUILD:
        qlabs.flush_receive()
    if verbose:
        print(f"  楼：{n_box} 栋整盒 + {n_cell} 格碎块（L 形楼），树 {n_tree} 棵"
              f"   [{'快速建图：不等确认' if FAST_BUILD else '逐件确认'}]")

    # 3) 起点 / 终点地砖。必须跟着 START_CELL / GOAL_CELL 走 ——
    #    写死角落坐标的话，起点一旦挪到路上，这块砖就会埋进楼里。
    for cell, col in ((START_CELL, [1.0, 0.15, 0.15]), (GOAL_CELL, [0.15, 0.35, 1.0])):
        city.create_building([cell[0] + 0.5, cell[1] + 0.5, 0.05], [0, 0, 0],
                             [1, 1, 0.1], col)

    # 4) 任务点标记：矮地砖 + 参考坐标系。
    #    场景里**没有任何文字标签**，只能靠颜色 + 标记来区分（见模块 docstring）。
    #    ⚠ 送货点**可以在楼顶**：这时地砖要画在楼顶面上，否则会埋在楼里看不见。
    def mark(cell, color, frame_h):
        i, j = tuple(cell)
        z = roof_top(i, j)                    # 空地 0 / 楼顶 该楼的高度
        city.create_building([i + 0.5, j + 0.5, z + 0.05], [0, 0, 0], [1, 1, 0.1], color)
        city.create_reference_frame([i + 0.5, j + 0.5, z + frame_h])

    if pickup:
        mark(pickup, PICKUP_COLOR, 1.6)
    if deliveries:
        for d in deliveries:
            mark(d, DELIVERY_COLOR, 1.2)

    # 5) 装饰层（只生成 actor，不影响 occupancy）
    #    路网装饰**永远都铺**（它是城市的一部分，不是可选装饰）；
    #    学生给的 DECOR 是**叠加**在它之上的额外件，不是替换。
    decorate(qlabs, city)
    if decor:
        n_dec = apply_decor(city, decor)
        if verbose:
            print(f"  装饰：路网 + 学生 DECOR 额外 {n_dec} 项")
    elif verbose:
        print("  装饰：只有内置路网")

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
        time.sleep(0.3)                      # 让最后一帧建图指令确认完成
        fleet = PedestrianFleet(city, n_street=6, n_cross=4, n_plaza=PLAZA_WALKERS)
        fleet.start()
        if verbose:
            print(f"  行人：{fleet.counts}（**会换班**：走完一趟就退休、换个新人在别处出现；"
                  f"无人机在 z=3.0、行人 ~1.8 m，实测不挡路）")
    elif verbose:
        print("  行人：off")

    return city


def decorate(qlabs, city):
    """路面与街道家具 —— 除树以外全是矮件（z ≤ 0.04），压在路上不会挡路

    **完全由街道剖面驱动**，没有一处写死坐标。
    z 分层（避免共面闪烁）：人行道 0.02 < 沥青 0.03 < 标线 0.04 < 家具。

    树**不在这里** —— 树是会挡路的，必须写进 ASCII（`T`）才进得了占据栅格。
    装饰层只放不挡路的东西，这条不变量是结构性的。
    """
    half = SIZE - 0.5
    ASPHALT = [0.13, 0.13, 0.145]
    PAVE = [0.62, 0.62, 0.64]
    PLAZA_PAVE = [0.72, 0.70, 0.66]
    rng = random.Random(CITY_SEED + 7)          # 装饰也固定种子，建出来的样子稳定

    xs, ys = sidewalk_lines()

    # 1) 人行道铺装
    for x in xs:
        city.create_road([[x, 0.5], [x, half]], width=SW, color=PAVE, z=0.02)
    for y in ys:
        city.create_road([[0.5, y], [half, y]], width=SW, color=PAVE, z=0.02)

    # 2) 广场铺装（比人行道亮一点的石板色，整块铺满）
    pi, pj = PLAZA_ORIGIN
    city.create_road([[pi + 0.5, pj + PLAZA_SIZE / 2], [pi + PLAZA_SIZE - 0.5, pj + PLAZA_SIZE / 2]],
                     width=PLAZA_SIZE, color=PLAZA_PAVE, z=0.018)

    # 3) 沥青：横竖两条车行道，贯穿整张图
    cx = CAR_LO + CARRIAGE / 2
    cy = CAR_LO + CARRIAGE / 2
    city.create_road([[cx, 0.5], [cx, half]], width=CARRIAGE, color=ASPHALT, z=0.03)
    city.create_road([[0.5, cy], [half, cy]], width=CARRIAGE, color=ASPHALT, z=0.03)

    # 4) 标线：中心黄线 + 两侧白色边线
    for x0, vertical in ((cx, True), (cy, False)):
        if vertical:
            city.create_road([[x0, 0.5], [x0, half]], width=0.14, color=[1.0, 0.80, 0.0], z=0.04)
            for off in (-CARRIAGE / 2 + 0.15, CARRIAGE / 2 - 0.15):
                city.create_road([[x0 + off, 0.5], [x0 + off, half]],
                                 width=0.10, color=[0.92, 0.92, 0.92], z=0.04)
        else:
            city.create_road([[0.5, x0], [half, x0]], width=0.14, color=[1.0, 0.80, 0.0], z=0.04)
            for off in (-CARRIAGE / 2 + 0.15, CARRIAGE / 2 - 0.15):
                city.create_road([[0.5, x0 + off], [half, x0 + off]],
                                 width=0.10, color=[0.92, 0.92, 0.92], z=0.04)

    # 5) 路口 4 条斑马线，跨满车行道
    for (c, d) in intersections():
        for loc, rot, k in crosswalk_spots(c, d):
            city.create_crosswalk(loc, rotation=rot, scale=[k, 1, 1])

    # 6) 街道家具：锥桶（施工围挡几簇）+ 路口四个角的参考坐标系
    for (ox, oy) in [(COR_LO - 1.5, COR_LO - 1.5), (COR_HI + 1.5, COR_LO - 1.5),
                     (COR_LO - 1.5, COR_HI + 1.5), (COR_HI + 1.5, COR_HI + 1.5)]:
        city.create_reference_frame([ox, oy, 1.6])

    for _ in range(6):                                  # 散落的施工锥
        x = rng.uniform(1.0, SIZE - 1.0)
        y = rng.uniform(1.0, SIZE - 1.0)
        if is_road(int(x), int(y)) or is_sidewalk(int(x), int(y)):
            city.create_cone([x, y, 0.0], [0, 0, 0], [1, 1, 1])

    # 广场中心摆一圈锥桶，当个"围栏/花坛"的暗示
    pcx, pcy = pi + PLAZA_SIZE / 2, pj + PLAZA_SIZE / 2
    for k in range(8):
        a = 2 * math.pi * k / 8
        city.create_cone([pcx + 2.2 * math.cos(a), pcy + 2.2 * math.sin(a), 0.0],
                         [0, 0, 0], [1.2, 1.2, 1.2])

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
    """场外郊区环 —— **全部落在 SIZE x SIZE 可飞区域之外**。

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

    # 一个角落放几栋高楼，做出"市中心天际线"。
    # ⚠ 坐标必须**由 SIZE 派生**：写死的话，SIZE 一变大这些楼就会落进可飞区内部，
    #   而 occupancy() 只由 ASCII 决定、不知道它们存在 -> A* 会规划穿楼路径，无人机必撞。
    for (di, dj, h) in [(1, 1, 9.0), (3, 1, 12.0), (1, 3, 7.0), (2, 2, 10.0)]:
        i, j = SIZE + di, SIZE + dj
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
    mhd = abs(START_CELL[0] - GOAL_CELL[0]) + abs(START_CELL[1] - GOAL_CELL[1])
    print(f"  最短路径 {steps} 步（曼哈顿下界 {mhd}）")
    print(f"  够不到的孤岛: {len(isolated)} 格 {'✓ 无' if not isolated else sorted(isolated)[:10]}")
    print(f"  死胡同 {len(de)} 个")
    return steps


if __name__ == "__main__":
    report()
