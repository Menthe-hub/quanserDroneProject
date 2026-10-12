# Quanser 无人机实验（QDrone2 + QLabs）

北航 Quanser 课程的实验代码。全部运行在 **Quanser Interactive Labs（QLabs）** 仿真器上，
用 Python 通过 `qvl` 库驱动 **QDrone2** 四旋翼。

---

## 三条实验线

| 线 | 入口 | 做什么 |
|---|---|---|
| **PID 控制** | `QuanserTask/qdrone2PID.py` | 用 PID 让无人机飞到指定点，统计误差分布 |
| **Q-Learning 走迷宫** | `Demo/QLearningSimulation.py` | 19×19 迷宫里从 `(0,0)` 飞到 `(18,18)`，用 Q-Learning 学策略、PID 执行 |
| **城市配送**（本仓新增） | `Demo/city_mission.py` | 在城市里取货 → 按自己排的顺序送货 → 到终点 |

Q-Learning 那一线的完整调参记录、统计检验与结论在 [`实验记录-QLearning.md`](实验记录-QLearning.md)。

---

## 环境要求

* Windows + 已安装并**加载了世界**的 QLabs
* Python 3.12
* `qvl` 库（随 QLabs 安装，本机在 `C:\Users\Mint\Documents\Quanser\libraries\python\qvl\`，
  需要把它加进 `PYTHONPATH` 或放进项目里）
* `numpy`、`matplotlib`、`opencv-python`、`pygame`（`--shot` 截图用）

> ⚠ **不要用外部 `timeout` 强杀连着 QLabs 的 Python 进程。** 客户端异常断开会让 QLabs
> 卸载世界，之后所有 `spawn` 静默失败。要限时就让它自己优雅退出（见下面的 `--seconds`）。

---

## 城市配送实验

```bash
python Demo/city_mission.py --check                    # 体检 + 任务检查（纯离线，秒级）
python Demo/city_mission.py --selftest                 # 自检：证明调度契约的检查会拦
python Demo/city_mission.py --build --seconds 30       # 建图，俯视相机看效果
python Demo/city_mission.py --build --shot 城市.jpg     # 建图并存一张俯视截图
python Demo/city_mission.py --run --virtual            # 纯网格虚拟跑一遍（不连 QLabs）
python Demo/city_mission.py --run                      # 真机跑一遍
python Demo/city_mission.py --seed 7                   # 换一个种子重新生成一座城
```

### 这一版的设计：从"寻路"变成"调度"

无人机巡航在 **z = 8.5 m**，比所有楼（4~7.8 m）都高。由此：

* **送货点可以摆在楼顶上** —— 飞到它正上方即可
* **障碍不构成约束**，两点之间直线可达 —— 寻路本身没有难度

于是学生要写的就只剩一件事：**`Module/StudentMap.py` 里的 `schedule()`**，
决定**按什么顺序送**。这是一个旅行商问题（TSP）味道的组合优化：

```python
def schedule(start, pickup, deliveries, goal):
    """返回送货点的访问顺序，必须是 0..n-1 的一个排列。"""
    return list(range(len(deliveries)))     # 默认：按文件顺序
```

当前这三个点，**照文件顺序送是 127.8 m，最短只要 102.2 m** —— 差 25 米。
文件里的 `DELIVERIES` 顺序是刻意打乱的，学生一上手就能看到改进空间。
`schedule()` 的 docstring 里点了三条路：暴力枚举、最近邻、2-opt。

任务运行时**每到一个点打一条提示**，不评分：

```
→ 前往送货点2 (6, 6)   直线 33.9 m
  ✓ 送达 2/3 （楼顶）
```

### 学生要改的只有两个文件

| 文件 | 改什么 |
|---|---|
| `Module/StudentMap.py` | 地图（ASCII）、任务点（起点/取货/送货/终点）、`schedule()` 算法 |
| — | 就这一个。`Demo/city_mission.py` 是执行器，一般不用动 |

地图是 `MapSpec.generate_city()` 按路网规则生成的 40×40 城市：

```
│← 人行道 2 →│← 车行道 4 →│← 人行道 2 →│   街道走廊 8 格
│←——————— 街区 16 格 ———————→│           四个街角，一个做成广场
```

正中间**一个十字路口**；广场里有散落的树、交通锥和会换班的行人；
街上有沿人行道走和过马路的行人（每人走完一趟就退休、换个新人在别处出现）。

---

## 文件结构

```
Module/
  MapSpec.py          地图层：路网生成、占据栅格、建图、任务检查（唯一真源）
  StudentMap.py       ★ 学生编辑面：地图 + 任务点 + schedule()
  City.py             QLabs 场景组件封装（楼/树/路面/行人线程…）
  QDrone2.py          真机 QDrone2 封装（PID 的 fly_to / takeoff）
  VirtualQDrone2.py   纯网格虚拟无人机，接口与真机一致
  Astar.py            A* 规划器（备用；当前设计飞在楼之上，用不到）
  QLabsLock.py        把 QLabs 访问串行化（多线程下必需）
  QLearning.py / Reward.py / Evaluator.py / Drawing.py / PIDController.py
Demo/
  city_mission.py     ★ 城市配送任务执行器
  city_map_demo.py    把地图灌进 QLearningSimulation 跑 RL（复用旧训练流程）
  QLearningSimulation.py / AstarSimulation.py / Maze.py
```

---

## 实测得出的几条硬约束

这些是踩出来的，写在代码注释里，也列在这里免得重复踩：

* **QLabs 的 Python 客户端不是线程安全的**：两个线程共用一个连接对象，会互相吃掉应答、
  甚至段错误。任何多线程访问都要先 `import Module.QLabsLock`。
  （不是"Python 线程不安全"，是**共享了一个不可共享的连接**。）
* **行人默认挡路，且没有任何接口能关掉**（`enable_collsion` 只管碰撞计数）。
  唯一的绕法是**飞在行人之上**。
* **建图 66.7 ms/块**，一座城就是几十秒 —— 只能建一次，绝不能放进逐帧循环。
  批量 spawn 不等确认（`waitForConfirmation=False`）可以把这个开销降到 0.02 ms，
  但**必须用 `spawn_id` 并自己编号**，否则材质/位姿会打到别的 actor 上。
* **`spawn_id` 的成功返回码是 `0`**（不是 `True`），写成 `if ok:` 会把成功判成失败。
* **`QLabsWalls`（泡沫板墙）无法做成障碍**，是纯视觉组件；真障碍只能用 `QLabsBasicShape`。
