import numpy as np

class QLearning:
    def __init__(self, action_size, SIZE, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.997, min_exploration=0.01):
        """
        输入变量：
            action_size: 动作空间大小
            SIZE: 地图大小
            learning_rate: 学习率
            discount_factor: 折扣因子
            exploration_rate: 探索率
            exploration_decay: 探索率衰减幅度
            min_exploration: 最小探索率
        """
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        
        # 离散化状态空间
        self.grid_size = SIZE
        self.q_table = np.zeros((self.grid_size, self.grid_size, action_size))
    
    def discretize_state(self, state):
        """将连续状态离散化为网格坐标
        输入变量：
            state: 当前状态
        """
        x = int(state[0])
        y = int(state[1])
        return x, y
    
    def act(self, state):
        """选择动作：ε-贪婪策略"""
        x, y = self.discretize_state(state)
        
        # 探索
        if np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_size)
        
        # 利用
        return np.argmax(self.q_table[x, y])
    
    def update(self, state, action, reward, next_state, done):
        """更新Q表
        输入变量：
            state: 当前状态
            action: 所选动作
            reward: 奖励值
            next_state: 下一个状态
            done: 是否完成本次训练
        """
        x, y = self.discretize_state(state)
        next_x, next_y = self.discretize_state(next_state)
        
        # 当前Q值
        current_q = self.q_table[x, y, action]
        
        if done:
            target = reward
        else:
            # 下一个状态的最大Q值
            max_next_q = np.max(self.q_table[next_x, next_y])
            target = reward + self.discount_factor * max_next_q
        
        # 更新Q值
        self.q_table[x, y, action] = current_q + self.learning_rate * (target - current_q)
    
    def decay_exploration(self):
        """衰减探索率"""
        self.exploration_rate = max(self.min_exploration, self.exploration_rate * self.exploration_decay)