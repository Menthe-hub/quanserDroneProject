"""A* 网格规划器

只依赖标准库（heapq）。**模块顶层不 import MapSpec** —— 既避免循环依赖
（MapSpec.mission_report 要用它），也让它能脱离 qvl / MapSpec 单独测试。

坐标约定与 `MapSpec.occupancy()` 一致：

    格子 (i, j)  ↔  世界坐标 (i+0.5, j+0.5)  ↔  占据栅格 grid[i+1, j+1]

栅格里 `0 = 墙`、`1 = 通行`。

四邻域、单位代价、曼哈顿启发式。曼哈顿距离在四邻域单位代价网格上**可采纳且一致**，
所以 A* 给出的路径长度与 BFS 相同（这正是可以用它交叉校验的原因）。
"""
import heapq


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _default_grid():
    """惰性取占据栅格，避免模块顶层的循环依赖"""
    from Module import MapSpec
    return MapSpec.occupancy()


def is_free(grid, cell):
    """格子是否可通行（越界算墙）"""
    i, j = cell
    gi, gj = i + 1, j + 1
    if gi < 0 or gj < 0 or gi >= grid.shape[0] or gj >= grid.shape[1]:
        return False
    return grid[gi, gj] == 1


def astar(start, goal, grid=None):
    """求最短路径。

    输入变量：
        start / goal: 格子坐标 (i, j)
        grid: (可选) 占据栅格；不给则取 MapSpec.occupancy()
    返回：
        格子列表 [(i,j), ...]（含起点与终点）；**无解或起终点落在墙上时返回 None**
    """
    if grid is None:
        grid = _default_grid()
    start, goal = tuple(start), tuple(goal)
    if not is_free(grid, start) or not is_free(grid, goal):
        return None
    if start == goal:
        return [start]

    # (f, 序号, 格)；序号保证 f 相同时先进先出，结果确定可复现
    heap = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    closed = set()
    counter = 1

    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]

        closed.add(cur)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cur[0] + di, cur[1] + dj)
            if nb in closed or not is_free(grid, nb):
                continue
            tentative = g_score[cur] + 1
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                came_from[nb] = cur
                heapq.heappush(heap, (tentative + _manhattan(nb, goal), counter, nb))
                counter += 1
    return None


def plan_route(start, waypoints, grid=None, goal=None):
    """按顺序经过一串途经点：start -> waypoints[0] -> ... -> goal

    输入变量：
        start: 起点格子
        waypoints: 有序途经点列表
        grid: (可选) 占据栅格
        goal: (可选) 终点；给了就作为最后一段
    返回 (cells, legs)：
        cells: 拼接好的完整格子路径（段与段之间去掉重复的衔接格）
        legs : [{"from":…, "to":…, "steps":n 或 None, "ok":bool}, …]
    **任一段无解时 cells 返回 None，legs 里标出是哪一段被堵死。**
    这本身就是运行期的"堵路测试"闸门 —— 学生把路堵上，这里自然就过不去。
    """
    if grid is None:
        grid = _default_grid()
    targets = list(waypoints) + ([goal] if goal is not None else [])

    cells = [tuple(start)]
    legs = []
    cur = tuple(start)
    for tgt in targets:
        tgt = tuple(tgt)
        seg = astar(cur, tgt, grid)
        if seg is None:
            legs.append({"from": cur, "to": tgt, "steps": None, "ok": False})
            return None, legs
        legs.append({"from": cur, "to": tgt, "steps": len(seg) - 1, "ok": True})
        cells.extend(seg[1:])          # seg[0] 就是上一段的终点，去掉避免重复
        cur = tgt
    return cells, legs


def path_to_world(cells, z=1.0):
    """格子路径 -> 世界坐标 setpoint 列表（格心），正好是 QDrone2.fly_to 要的格式"""
    return [[i + 0.5, j + 0.5, z] for (i, j) in cells]
