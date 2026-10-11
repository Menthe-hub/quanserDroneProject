import numpy as np


class VirtualQDrone2:
    """纯网格虚拟无人机：对外接口与 Module/QDrone2.py 的 QDrone2 完全一致，但不连接 QLabs。

    用途：QLearningSimulation.py 加 --virtual 后用它替代 QDrone2

    墙体由 MAZE_POINTS 按与 build_city() 完全相同的方式生成
    """

    def __init__(self, maze_points, size):
        """
        输入变量：
            maze_points: 迷宫障碍物坐标列表（与 QLearningSimulation.MAZE_POINTS 同一份）
            size: 地图边长
        """
        self.size = size
        self.location = [0.5, 0.5, 0.0]
        self.collision = False

        # 与 build_city() 里的 create_maze() 保持完全一致的写法
        self.maze = np.ones((size + 2, size + 2))
        for i in range(size + 2):
            self.maze[0, i] = 0
            self.maze[i, 0] = 0
            self.maze[size + 1, i] = 0
            self.maze[i, size + 1] = 0
        for point in maze_points:
            self.maze[int(point[0] + 0.5), int(point[1] + 0.5)] = 0

    def _is_wall(self, location):
        """判断某个坐标是否落在障碍格里（越界也算障碍）"""
        i = int(location[0] + 0.5)
        j = int(location[1] + 0.5)
        if i < 0 or j < 0 or i > self.size + 1 or j > self.size + 1:
            return True
        return self.maze[i, j] == 0

    def create_qdrone2(self, location, rotation, scale, actorNumber=0):
        """创建 QDrone2 无人机（虚拟版只记录位置）"""
        self.location = list(location)
        self.collision = False
        return True

    def destroy_all_qdrone2(self):
        """清除地图中所有 QDrone2 无人机（虚拟版无需操作）"""
        pass

    def get_location(self, moterEnabled=True):
        """获取 QDrone2 无人机状态"""
        return True, list(self.location), [0.0, 0.0, 0.0], 0.0, self.collision

    def takeoff(self, setpoint, z_error=0.05, moterEnabled=True):
        """控制 QDrone2 无人机起飞（虚拟版直接置为悬停高度）"""
        self.location = [self.location[0], self.location[1], setpoint[2]]
        self.collision = False
        return True

    def fly_to(self, setpoint, desired_orientation=[0, 0, 0], error=0.05, moterEnabled=True):
        """飞向定点。

        与真机一致：以目标所在的格子判定是否撞墙。
        撞墙时返回的 location 是"目标点"而不是实际停下的位置——这不影响训练，
        因为碰撞会立刻终止本回合，该位置不会被用于任何后续决策。
        """
        target = list(setpoint)
        if self._is_wall(target):
            self.collision = True
            return True, target, True
        self.location = target
        self.collision = False
        return True, list(self.location), False

    def land(self, moterEnabled=False):
        """控制 QDrone2 无人机降落（虚拟版无需操作）"""
        pass
