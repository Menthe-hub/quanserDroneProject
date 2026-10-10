import math
import numpy as np
import sys
from pathlib import Path
from qvl.qlabs import QuanserInteractiveLabs

dir=Path(__file__).parent.parent
if dir not in sys.path:
    sys.path.append(str(dir))

from Module.City import City
from Module.QDrone2 import QDrone2
from Module.Drawing import Draw
from Module.Evaluator import Evaluator

SIZE=19
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

def display_city():
    print("\n=== 地图示意图 ===")
    print(f"坐标系说明:\n起点G为红色, 坐标为(0,0)\n终点S为绿色, 坐标为({SIZE},{SIZE})")
    print("\n地图网格 (■ = 障碍物, □ = 可通行, G = 起点, S = 终点):")
    for j in range(SIZE+2):
        for i in range(SIZE+2):
            if i==1 and j==SIZE:
                print(" G ", end="")
            elif i==SIZE and j==1:
                print(" S ", end="")
            elif MAZE[i,SIZE+1-j]==0:
                print(" ■ ", end="")
            else:
                print(" □ ", end="")
        print()

def get_user_point():
    while True:
        user_input=input("请输入坐标点 x,y 或 'done' 结束: ").strip()
        
        if user_input.lower()=='done':
            return None
        
        if user_input.lower()=='help':
            print("\n输入格式说明:")
            print("  - 坐标点格式: x,y")
            print("  - 示例: 1,2")
            print(f"  - 范围: x和y都应在0到{SIZE}之间")
            print("  - 结束输入: 输入 'done'")
            print("  - 帮助: 输入 'help'")
            continue
        
        try:
            # 解析输入
            parts=user_input.split(',')
            if len(parts)!=2:
                print("错误: 请输入两个数字, 用英文逗号分隔")
                continue
                
            x=int(parts[0])
            y=int(parts[1])
            
            if x<0 or x>SIZE or y<0 or y>SIZE:
                print(f"错误: 坐标值应在0到{SIZE}之间")
                continue
                
            return (x, y)
            
        except ValueError:
            print("错误: 请输入有效的数字")

def user_to_world(x_user, y_user):
    world_x=x_user+0.5
    world_y=y_user+0.5
    return world_x,world_y

if __name__=="__main__":
    # 连接QLabs
    qlabs=QuanserInteractiveLabs()
    if not qlabs.open("localhost"):
        print("无法连接QLabs")
    
    # 构建环境
    build_city(qlabs)
    display_city()
    evaluator = Evaluator()
    
    # 创建QDrone2无人机实例
    qdrone2=QDrone2(qlabs)
    qdrone2.destroy_all_qdrone2()
    create_success=qdrone2.create_qdrone2([0.5,0.5,0.35], [0,0,0], [0.25,0.25,0.25])
    if create_success:
        evaluator.start_timer()
        takeoff_success=qdrone2.takeoff([0.5,0.5,1]) # 起飞
        if takeoff_success:
            success, location, _, _, _=qdrone2.get_location()
            evaluator.add_waypoint((0.5,0.5),(location[0], location[1]))
        else:
            print("QDrone2无人机起飞失败")
    else:
        print("无法创建QDrone2无人机实例")
        
    print("\n=== 输入说明 ===")
    print("请输入无人机需要到达的坐标点")
    print("格式: x,y (例如: 1,2)")
    print("注意: 使用英文逗号分隔，不要有空格")
    print("输入 'done' 结束输入")
    print("输入 'help' 显示帮助信息")
    print("================\n")
    
    x=[0]
    y=[0]
    finished=False
    last_location=[]
    while True:
        point = get_user_point()
        if point is None:
            break
        
        x_user,y_user=point
        x.append(x_user)
        y.append(y_user)
        print(f"已添加点: ({x_user},{y_user})")
        
        # 飞向该点
        world_x,world_y=user_to_world(x_user, y_user)
        success,location,collision=qdrone2.fly_to((world_x,world_y,1))
        evaluator.add_waypoint((world_x, world_y),(location[0], location[1]))
        last_location = location
        if not success:
            print("飞行失败，程序结束")
            break
        if collision:
            print(f"警告: 无人机与障碍物碰撞!")
            break
        
    evaluator.stop_timer()
    qdrone2.land()
    if not x and not y:
        print("未输入任何点，程序结束")
        qlabs.close()
    
    if math.sqrt((last_location[0]-GOAL[0])**2+(last_location[1]-GOAL[1])**2)<GOAL_TOLERANCE:
        finished=True
        print("\n无人机已成功到达目标点!")
    
    print(f"\n您输入的路径点: {list(zip(x, y))}")
    x=np.array(x)+0.5
    y=np.array(y)+0.5
    drawer=Draw(SIZE, MAZE)
    drawer.draw(x, y, "Trajectory", "X", "Y", show_MAZE=True, trajectory=False)

    print(evaluator.generate_evaluation_report(finished))
    evaluator.plot_performance_analysis()