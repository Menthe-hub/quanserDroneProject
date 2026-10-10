import math
import numpy as np
import sys
from tqdm import tqdm
from pathlib import Path
from qvl.qlabs import QuanserInteractiveLabs

dir=Path(__file__).parent.parent
if dir not in sys.path:
    sys.path.append(str(dir))

from Module.City import City
from Module.QDrone2 import QDrone2
from Module.QLearning import QLearning
from Module.Reward import Reward
from Module.Drawing import Draw
from Module.Evaluator import Evaluator

SIZE=19
EPISODES=2000
MAX_STEPS=100
GOAL_TOLERANCE=0.3
ORIGINE=[0.5, 0.5]
GOAL=[SIZE-0.5, SIZE-0.5]

MAZE=np.ones((SIZE+2,SIZE+2))
for i in range(SIZE+2):
    MAZE[0,i]=0
    MAZE[i,0]=0
    MAZE[SIZE+1,i]=0
    MAZE[i,SIZE+1]=0

MAZE_POINTS=[
        [1.5,0.5,1.5],[7.5,0.5,1.5],[13.5,0.5,1.5],[15.5,0.5,1.5],
        [1.5,1.5,1.5],[2.5,1.5,1.5],[3.5,1.5,1.5],[4.5,1.5,1.5],[5.5,1.5,1.5],[7.5,1.5,1.5],[9.5,1.5,1.5],[10.5,1.5,1.5],[11.5,1.5,1.5],[11.5,1.5,1.5],[12.5,1.5,1.5],[13.5,1.5,1.5],[15.5,1.5,1.5],[16.5,1.5,1.5],[17.5,1.5,1.5],
        [7.5,2.5,1.5],[11.5,2.5,1.5],
        [1.5,3.5,1.5],[2.5,3.5,1.5],[3.5,3.5,1.5],[4.5,3.5,1.5],[5.5,3.5,1.5],[7.5,3.5,1.5],[9.5,3.5,1.5],[11.5,3.5,1.5],[13.5,3.5,1.5],[14.5,3.5,1.5],[15.5,3.5,1.5],[17.5,3.5,1.5],
        [1.5,4.5,1.5],[7.5,4.5,1.5],[9.5,4.5,1.5],[13.5,4.5,1.5],[15.5,4.5,1.5],[17.5,4.5,1.5],
        [1.5,5.5,1.5],[3.5,5.5,1.5],[4.5,5.5,1.5],[5.5,5.5,1.5],[6.5,5.5,1.5],[7.5,5.5,1.5],[8.5,5.5,1.5],[9.5,5.5,1.5],[11.5,5.5,1.5],[12.5,5.5,1.5],[13.5,5.5,1.5],[15.5,5.5,1.5],[17.5,5.5,1.5],[18.5,5.5,1.5],
        [1.5,6.5,1.5],[5.5,6.5,1.5],[11.5,6.5,1.5],[15.5,6.5,1.5],
        [0.5,7.5,1.5],[1.5,7.5,1.5],[3.5,7.5,1.5],[4.5,7.5,1.5],[5.5,7.5,1.5],[7.5,7.5,1.5],[9.5,7.5,1.5],[11.5,7.5,1.5],[13.5,7.5,1.5],[15.5,7.5,1.5],[16.5,7.5,1.5],[17.5,7.5,1.5],
        [3.5,8.5,1.5],[7.5,8.5,1.5],[9.5,8.5,1.5],[11.5,8.5,1.5],[13.5,8.5,1.5],[17.5,8.5,1.5],
        [1.5,9.5,1.5],[3.5,9.5,1.5],[4.5,9.5,1.5],[5.5,9.5,1.5],[6.5,9.5,1.5],[7.5,9.5,1.5],[9.5,9.5,1.5],[11.5,9.5,1.5],[12.5,9.5,1.5],[13.5,9.5,1.5],[15.5,9.5,1.5],[16.5,9.5,1.5],[17.5,9.5,1.5],
        [1.5,10.5,1.5],[9.5,10.5,1.5],[15.5,10.5,1.5],
        [1.5,11.5,1.5],[2.5,11.5,1.5],[3.5,11.5,1.5],[4.5,11.5,1.5],[5.5,11.5,1.5],[6.5,11.5,1.5],[7.5,11.5,1.5],[9.5,11.5,1.5],[11.5,11.5,1.5],[12.5,11.5,1.5],[13.5,11.5,1.5],[15.5,11.5,1.5],[17.5,11.5,1.5],
        [1.5,12.5,1.5],[5.5,12.5,1.5],[7.5,12.5,1.5],[9.5,12.5,1.5],[13.5,12.5,1.5],[15.5,12.5,1.5],[17.5,12.5,1.5],
        [0.5,13.5,1.5],[1.5,13.5,1.5],[3.5,13.5,1.5],[5.5,13.5,1.5],[7.5,13.5,1.5],[9.5,13.5,1.5],[10.5,13.5,1.5],[11.5,13.5,1.5],[13.5,13.5,1.5],[15.5,13.5,1.5],[17.5,13.5,1.5],[18.5,13.5,1.5],
        [3.5,14.5,1.5],[7.5,14.5,1.5],[9.5,14.5,1.5],[11.5,14.5,1.5],[13.5,14.5,1.5],[17.5,14.5,1.5],
        [1.5,15.5,1.5],[2.5,15.5,1.5],[3.5,15.5,1.5],[5.5,15.5,1.5],[7.5,15.5,1.5],[8.5,15.5,1.5],[9.5,15.5,1.5],[11.5,15.5,1.5],[13.5,15.5,1.5],[15.5,15.5,1.5],[16.5,15.5,1.5],[17.5,15.5,1.5],
        [1.5,16.5,1.5],[5.5,16.5,1.5],[11.5,16.5,1.5],[13.5,16.5,1.5],[17.5,16.5,1.5],
        [0.5,17.5,1.5],[1.5,17.5,1.5],[3.5,17.5,1.5],[4.5,17.5,1.5],[5.5,17.5,1.5],[6.5,17.5,1.5],[7.5,17.5,1.5],[9.5,17.5,1.5],[11.5,17.5,1.5],[13.5,17.5,1.5],[14.5,17.5,1.5],[15.5,17.5,1.5],[17.5,17.5,1.5],
        [7.5,18.5,1.5],[9.5,18.5,1.5],[15.5,18.5,1.5]
    ]

def create_maze(point):
    MAZE[int(point[0]+0.5), int(point[1]+0.5)]=0

def build_city(qlabs):

    city_construction=City(qlabs)
    city_construction.clean_city()
    
    # 构建边界
    city_construction.create_building([-0.5,SIZE/2,10.5], [0,0,0], [1,SIZE+2,21], [0.1,0.1,0.1], roughness=0.01)
    city_construction.create_building([SIZE/2,-0.5,10.5], [0,0,0], [SIZE,1,21], [0.1,0.1,0.1], roughness=0.01)
    city_construction.create_building([SIZE+0.5,SIZE/2,10.5], [0,0,0], [1,SIZE+2,21], [0.1,0.1,0.1], roughness=0.01)
    city_construction.create_building([SIZE/2,SIZE+0.5,10.5], [0,0,0], [SIZE,1,21], [0.1,0.1,0.1], roughness=0.01)
    
    # 搭建迷宫
    for i in range(len(MAZE_POINTS)):
        city_construction.create_building(MAZE_POINTS[i], [0,0,0], [1,1,3], [0.3,0.3,0.3])
        create_maze(MAZE_POINTS[i])

    # 构建起点和终点
    city_construction.create_building([0.5,0.5,0.05], [0,0,0], [1,1,0.1], [1,0,0])# 起点
    city_construction.create_building([SIZE-0.5,SIZE-0.5,0.05], [0,0,0], [1,1,0.1], [0,1,0])# 终点

def reset_environment(qlabs):
    """重置环境"""
    qdrone2=QDrone2(qlabs)
    qdrone2.destroy_all_qdrone2()
    create_success=qdrone2.create_qdrone2([0.5,0.5,0.35], [0,0,0], [0.25,0.25,0.25])
    if create_success:
        takeoff_success=qdrone2.takeoff([0.5,0.5,1])
        if takeoff_success:
            success, location, _, _, collision=qdrone2.get_location()
            return qdrone2, success, location, collision
        else:
            return False, False, False, False
    else:
        return False, False, False, False

def execute_action(qdrone2, location, action):
    """执行动作并返回新状态
    输入变量：
        location: 当前位置
        action: 所选动作
        action_space_type: 动作类型
    """
    setpoint=[]
    action_mapping={
        0: [1, 0, 0],
        1: [-1, 0, 0],
        2: [0, 1, 0],
        3: [0, -1, 0],
    }
    for i in range(3):
        setpoint.append(location[i]+action_mapping.get(action, [0, 0, 0])[i])
    
    fly_success, location, collision=qdrone2.fly_to(setpoint)
    return fly_success, location, collision, setpoint
    
def if_done(location, step, collision):
    """检查是否结束当前回合
    输入变量：
        location: 当前位置
        step: 当前步数
        collision: 是否碰撞
    """
    if collision:
        return True, True, False
    
    if step>MAX_STEPS:
        return True, False, False
    
    distance_to_goal=math.dist(GOAL, [location[0], location[1]])
    if distance_to_goal<GOAL_TOLERANCE:
        return True, True, True
    
    return False, False, False

def qlearning_train(qlabs,drawer):
    """使用Q-learning算法训练无人机"""
    print("开始Q-learning训练...")
    
    # 初始化Q-learning
    action_size=4
    agent=QLearning(action_size, SIZE)
    success_count=0
    total_rewards=[]
    total_episode=[]
    reward_calculator=Reward(EPISODES, MAX_STEPS, ORIGINE, GOAL)
    
    for episode in tqdm(range(EPISODES)):
        # 重置环境
        qdrone2, success, prev_location, collision=reset_environment(qlabs)
        if not success:
            print(f"回合{episode+1}无法初始化QDrone2无人机")
            continue
        
        total_reward=0
        done=False
        step=0
        prev_action=None
        
        while not done and step < MAX_STEPS:
            # 选择动作
            current_action=agent.act(prev_location)
            success, next_location, collision, _=execute_action(qdrone2, prev_location, current_action)
            if not success:
                print(f"回合{episode+1}动作执行失败")
            
            done, finish, not_collision=if_done(next_location, step, collision)
            step_reward=reward_calculator.reward_calculate(
                next_location, prev_action, current_action, step, done, finish, not_collision, episode
            )
            total_reward+=step_reward
            
            # 更新Q表
            agent.update(prev_location, current_action, step_reward, next_location, done)
            
            # 状态转移
            prev_location=next_location
            prev_action=current_action
            step+=1
        
        # 衰减探索率
        agent.decay_exploration()
        
        # 更新统计
        total_rewards.append(total_reward)
        total_episode.append(episode)

        # 检查是否成功到达目标
        if finish and not_collision:
            success_count+=1
        
        # 每100回合输出一次统计信息
        if (episode+1)%100==0:
            avg_reward=np.mean(total_rewards[-100:])
            print(f"\n=== Q-learning统计（回合{episode-99}-{episode+1}）===")
            print(f"成功率: {success_count:.1f}%")
            print(f"平均奖励: {avg_reward:.2f}")
            success_count = 0
    
    print("Q-learning训练完成!")
    drawer.draw(total_episode, total_rewards, "Reward Line Graph", "Episodes", "Rewards", show_MAZE=False, trajectory=False)
    return agent

def qlearning_test(qlabs, agent, evaluator):
    """测试训练好的Q-learning智能体"""
    # 重置环境
    evaluator.start_timer()
    qdrone2, success, location, _=reset_environment(qlabs)
    if not success:
        return None, None, False
    
    win=False
    done=False
    step=0
    x=[location[0]]
    y=[location[1]]
    evaluator.add_waypoint((0.5,0.5),(location[0], location[1]))
    # 保存原始探索率
    original_exploration=agent.exploration_rate
    agent.exploration_rate=0  # 测试时完全利用
    
    while not done:
        # 选择动作
        current_action=agent.act(location)
        success, next_location, collision, setpoint=execute_action(qdrone2, location, current_action)
        evaluator.add_waypoint((setpoint[0], setpoint[1]), (next_location[0], next_location[1]))
        if not success:
            print("测试过程中动作执行失败")
            break
        
        done, finish, not_collision=if_done(next_location, step, collision)
        
        # 记录路径
        x.append(next_location[0])
        y.append(next_location[1])

        # 转移到下一个状态
        location=next_location
        step+=1
    
    evaluator.stop_timer()
    # 恢复探索率
    agent.exploration_rate=original_exploration
    
    # 显示测试结果
    if finish and not_collision:
        win=True
        print(f"Q-learning测试成功! 用时 {step} 步到达目标")
    elif finish and not not_collision:
        print(f"Q-learning测试失败! 与障碍物发生碰撞")
    else:
        print(f"Q-learning测试失败! 未能在{MAX_STEPS}步内到达目标")
    
    return x, y, win

if __name__=="__main__":
    # 连接QLabs
    qlabs=QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("无法连接QLabs")
    
    # 构建环境
    build_city(qlabs)
    
    evaluator = Evaluator()
    
    # 开始训练
    drawer=Draw(SIZE, MAZE)
    agent=qlearning_train(qlabs, drawer)
    x,y,finished=qlearning_test(qlabs, agent, evaluator)
    drawer.draw(x, y, "Trajectory", "x", "y")
    
    print(evaluator.generate_evaluation_report(finished))
    evaluator.plot_performance_analysis()