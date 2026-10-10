import matplotlib.pyplot as plt

class Draw:
    def __init__(self, SIZE, MAZE):
        """
        输入变量：
            SIZE: 迷宫大小
            MAZE: 迷宫地图
        """
        self.SIZE=SIZE
        self.MAZE=MAZE
        
    def draw(self, x, y, title, xlabel, ylabel, show_MAZE=True, trajectory=True):
        """绘制轨迹曲线
        输入变量：
            x: x坐标列表或数组
            y: y坐标列表或数组
            title: 图表标题
            xlabel: x轴标签
            ylabel: y轴标签
            show_MAZE: 是否显示迷宫
            trajectory: 是否突出显示起点和终点
        """
        if show_MAZE:
            plt.figure(figsize=(self.SIZE+2, self.SIZE+2))
            for i in range(self.MAZE.shape[0]):
                for j in range(self.MAZE.shape[1]):
                    if self.MAZE[i, j]==0:
                        plt.fill_between([i-1, i], j-1, j, color='gray', alpha=0.5)
            plt.fill_between([0, 1], 0, 1, color='r', alpha=0.5)
            plt.fill_between([self.SIZE-1, self.SIZE], self.SIZE-1, self.SIZE, color='g', alpha=0.5)
        else:
            plt.figure(figsize=(10, 8))

        plt.plot(x, y, marker='o', color='b', linestyle='-', linewidth=2)
        if trajectory:
            plt.plot(x[0],y[0], marker='o', color='r', markersize=10)
            plt.plot(x[-1],y[-1], marker='o', color='g', markersize=10)
        plt.title(title, fontsize=14)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()