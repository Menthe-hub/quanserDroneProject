class Reward:
    def __init__(self, EPISODES, MAX_STEPS, ORIGINE, GOAL):
        """
        输入变量:
            EPISODES: 训练次数
            MAX_STEPS: 最大步数
            ORIGINE : 起点
            GOAL : 终点
        """
        self.prev_location=[]
        self.EPISODES=EPISODES
        self.MAX_STEPS=MAX_STEPS
        self.ORIGINE=ORIGINE
        self.GOAL=GOAL
    
    def mhd_dist(self, point1, point2):
        """计算曼哈顿距离"""
        dist=point1[0]-point2[0]+point1[1]-point2[1]
        if dist<0:
            dist=-dist
        return dist
    
    def __success(self, done, finish, not_collision, step):
        """计算到达奖励
        输入变量：
            done: 本次训练结束与否
            finish: 是否成功到达终点
            not_collision: 是否与障碍物发生碰撞
            step: 当前步数
        """
        reward=0
        if done:
            if finish and not_collision:
                reward += 1000+(self.MAX_STEPS-step)*0.5
                return reward
            elif finish and not not_collision:
                reward -= 1300
                return reward
            else:
                reward -= 1200
                return reward
        return reward
    
    def __explore(self, current_location):
        """阶段一：探索地图
        输入变量：
            current_location: 当前位置
        """
        reward=0
        current_location=[int(current_location[0]),int(current_location[1])]
        if current_location not in self.prev_location:
            self.prev_location.append(current_location)
            reward+=30
        else:
            reward-=10
        return reward
    
    def __arrive(self, current_location):
        """阶段二：到达终点
        输入变量：
            current_location: 当前位置
        """
        reward=0
        # 距离目标的进度奖励
        current_distance = self.mhd_dist(self.GOAL, [current_location[0], current_location[1]])
        max_distance = self.mhd_dist(self.GOAL,self.ORIGINE)
        if current_distance < max_distance/9:
            reward+=10
        elif current_distance < max_distance*2/9:
            reward+=6
        elif current_distance < max_distance/3:
            reward+=2
        elif current_distance < max_distance*4/9:
            reward-=1
        elif current_distance < max_distance*5/9:
            reward-=4
        elif current_distance < max_distance*2/3:
            reward-=6
        elif current_distance < max_distance*7/9:
            reward-=8
        elif current_distance < max_distance*8/9:
            reward-=9
        else:
            reward-=10
        return reward

    def __flight_back(self, prev_action, current_action):
        """惩罚回头路
        输入变量：
            prev_action: 前次动作
            current_action: 当前动作
        """
        reward=0
        if prev_action!=None:
            if prev_action+current_action==1 or prev_action+current_action==5:
                reward-=60
        return reward

    def reward_calculate(self, current_location, prev_action, current_action, step, done, finish, not_collision, episode):
        """计算总奖励
        输入变量：
            current_location: 当前位置
            prev_action: 前次动作
            current_action: 当前动作
            tep: 当前步数
            done: 本次训练结束与否
            finish: 是否成功到达终点
            not_collision: 是否与障碍物发生碰撞
            episode: 当前训练次数
        """
        if done:
            reward=self.__success(done, finish, not_collision, step)
        elif episode<=self.EPISODES/3:
            reward=self.__explore(current_location)
        else:
            reward=self.__arrive(current_location)
            reward=self.__flight_back(prev_action, current_action)
        return reward