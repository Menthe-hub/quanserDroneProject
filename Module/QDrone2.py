import sys
import math
import time
from pathlib import Path
from qvl.qdrone2 import QLabsQDrone2

dir=Path(__file__).parent.parent
if dir not in sys.path:
    sys.path.append(str(dir))

from Module.PIDController import PIDController

class QDrone2:
    def __init__(self,qlabs):
        self.qdrone2=QLabsQDrone2(qlabs)
    
    def create_qdrone2(self, location, rotation, scale, actorNumber=0):
        """创建QDrone2无人机
        输入变量：
            location: 几何中心的位置
            rotaation: 方向
            scale: 大小
            actorNumber: 编队序号
        """
        false=self.qdrone2.spawn_id(
            actorNumber=actorNumber,
            location=location,
            rotation=rotation,
            scale=scale
        )
        if not false:
            return True
        else:
            return False
    
    def destroy_all_qdrone2(self):
        """清除地图中所有QDrone2无人机"""
        self.qdrone2.destroy_all_actors_of_class()
    
    def get_location(self, moterEnabled=True):
        """获取QDrone2无人机状态
        输入变量：
            moterEnabled: 是否启动电机
        """
        success, location, orientation, _, _, TOFDistance, collision, _, _= self.qdrone2.set_velocity_and_request_state(
            motorsEnabled=moterEnabled,
            velocity=[0, 0, 0],
            orientation=[0, 0, 0]
        )
        return success, location, orientation, TOFDistance, collision
    
    def takeoff(self, setpoint, z_error=0.05, moterEnabled=True):
        """控制QDrone2无人机起飞
        输入变量：
            setpoint: 目标点
            z_error: 允许的最大误差
            moterEnabled: 是否启动电机
        """
        success, prev_location, _, _, _=self.get_location()
        if not success:
            return False
        z_pid=PIDController(6,0.8,0.5,setpoint[2])
        while True:
            vz=z_pid.update(prev_location[2])
            _, location, _, _, _, _, collision, _, _=self.qdrone2.set_velocity_and_request_state(
                motorsEnabled=moterEnabled,
                velocity=[0, 0, vz],
                orientation=[0, 0, 0]
            )
            if collision:
                return False
            if math.dist(location, setpoint)<z_error:
                self.qdrone2.set_velocity_and_request_state(
                    motorsEnabled=moterEnabled,
                    velocity=[0, 0, 0],
                    orientation=[0, 0, 0]
                )
                return True
            prev_location=location
            time.sleep(0.01)
            
    def land(self, moterEnabled=False):
        """控制QDrone2无人机降落
        输入变量：
            moterEnabled: 是否启动电机
        """
        self.qdrone2.set_velocity_and_request_state(
            motorsEnabled=moterEnabled,
            velocity=[0, 0, 0],
            orientation=[0, 0, 0]
        )
        time.sleep(0.1)
        
    def fly_to(self, setpoint, desired_orientation=[0,0,0], error=0.05, moterEnabled=True):
        """控制QDrone2无人机保持固定姿态并飞向固定点
        输入变量：
            setpoint: 目标点
            desired_orientation: 目标姿态
            error: 允许的最大误差
            moterEnabled: 是否启动电机
        """
        success, prev_location, _, _, collision=self.get_location()
        if not success:
            return success, prev_location, collision
        # 增益由 (2,0.8,0.1) 提到 (6,0.8,0.5)：1m 移动 0.67s -> 0.41s（实测 0.35~0.41）
        # 注意：这个增益只有在 execute_action 以格心为目标（误差不累积）时才是安全的，
        # 用"当前位置+1"的旧写法会贴墙蹭墙。
        x_pid=PIDController(6,0.8,0.5,setpoint[0])
        y_pid=PIDController(6,0.8,0.5,setpoint[1])
        z_pid=PIDController(6,0.8,0.5,setpoint[2])
        while True:
            vx=x_pid.update(prev_location[0])
            vy=y_pid.update(prev_location[1])
            vz=z_pid.update(prev_location[2])
            success, location, _, _, _, _, collision, _, _=self.qdrone2.set_velocity_and_request_state(
                motorsEnabled=moterEnabled,
                velocity=[vx, vy, vz],
                orientation=desired_orientation
            )
            if collision:
                return success, location, collision
            if math.dist(location, setpoint)<error:
                success, location, _, _, _, _, collision, _, _=self.qdrone2.set_velocity_and_request_state(
                    motorsEnabled=moterEnabled,
                    velocity=[0, 0, 0],
                    orientation=[0, 0, 0]
                )
                return success, location, collision
            prev_location=location
            time.sleep(0.01)