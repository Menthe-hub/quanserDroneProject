import time
import math
import random
import threading
import traceback
from qvl.basic_shape import QLabsBasicShape
from qvl.person import QLabsPerson
from qvl.character import QLabsCharacter
from qvl.traffic_cone import QLabsTrafficCone
from qvl.stop_sign import QLabsStopSign

class City:
    def __init__(self, qlabs):
        self.qlabs=qlabs
    
    def clean_city(self):
        """清空原有建筑物"""
        QLabsBasicShape(self.qlabs).destroy_all_actors_of_class()
        QLabsPerson(self.qlabs).destroy_all_actors_of_class()
        QLabsCharacter(self.qlabs).destroy_all_actors_of_class()
        QLabsTrafficCone(self.qlabs).destroy_all_actors_of_class()
        QLabsStopSign(self.qlabs).destroy_all_actors_of_class()
        
    def create_building(self, location, rotation, scale, color, configuration=QLabsBasicShape.SHAPE_CUBE, roughness=0.8):
        """创建建筑物
        输入变量：
            location: 几何中心的位置
            rotaation: 方向
            scale: 大小
            color: 颜色
            configuration: 形状，有SHAPE_CUBE、SHAPE_CYLINDER、SHAPE_SPHERE、SHAPE_CONE
            roughness: 粗糙度
        """
        building = QLabsBasicShape(self.qlabs)
        building.spawn(
            location=location,
            rotation=rotation,
            scale=scale,
            configuration=configuration
        )
        building.set_material_properties(color=color, roughness=roughness)
        return building
        
    def create_cone(self, location, rotation, scale):
        """创建交通锥
        输入变量：
            location: 几何中心的位置
            rotaation: 方向
            scale: 大小
        """
        cone = QLabsTrafficCone(self.qlabs)
        cone.spawn(
            location=location,
            rotation=rotation,
            scale=scale
        )
        return cone
        
    def create_stop_sign(self, location, rotation, scale):
        """创建停止标志
        输入变量：
            location: 几何中心的位置
            rotaation: 方向
            scale: 大小
        """
        stop_sign = QLabsStopSign(self.qlabs)
        stop_sign.spawn(
            location=location,
            rotation=rotation,
            scale=scale
        )
        return stop_sign
        
    def create_character(self, location, rotation, scale):
        """创建行人
        输入变量：
            location: 几何中心的位置
            rotaation: 方向
            scale: 大小
        """
        character = QLabsPerson(self.qlabs)
        character.spawn(
            location=location,
            rotation=rotation,
            scale=scale
        )
        return character
    
    def move_to_position(self, character, point, speed):
        """直线移动到指定点
        输入变量：
            character: 行人类
            point: 目标点
            speed: 移动速度
        """
        character.move_to(point, speed)
    
    def periodic_move(self, character, location, waypoints, speed):
        """折线往复运动
        输入变量：
            character: 行人类
            location: 行人初始位置
            waypoints: 运动过程中所有的途经点
            speed: 移动速度
        """
        try:
            n=0
            prev_point=location
            while True:
                m=n%len(waypoints)
                character.move_to(waypoints[m], speed)
                time.sleep(math.dist(waypoints[m], prev_point)/speed+1)
                n+=1
                prev_point=waypoints[m]
        except Exception as e:
            print(f"线程发生错误: {e}")
            traceback.print_exc()
    
    def periodic_thread(self, character, location, waypoints, speed):
        """启动折线往复运动线程
        输入变量：
            character: 行人类
            location: 行人初始位置
            waypoints: 运动过程中所有的途经点
            speed: 移动速度
        """
        try:
            thread=threading.Thread(target=self.periodic_move, args=(character, location, waypoints, speed))
            thread.start()
        except Exception as e:
            print(f"线程发生错误: {e}")
            traceback.print_exc()
    def random_move(self, character, location, waypoints, speed):
        """布朗运动
        输入变量：
            character: 行人类
            location: 行人初始位置
            waypoints: 运动过程中所有的途经点
            speed: 移动速度
        """
        try:
            prev_point=location
            while True:
                point=random.sample(waypoints,1)[0]
                character.move_to(point, speed)
                time.sleep(math.dist(point, prev_point)/speed+1)
                prev_point=point
        except Exception as e:
            print(f"线程发生错误: {e}")
            traceback.print_exc()
    
    def random_thread(self, character, location, waypoints, speed):
        """启动布朗运动线程
        输入变量：
            character: 行人类
            location: 行人初始位置
            waypoints: 运动过程中所有的途经点
            speed: 移动速度
        """
        try:
            thread=threading.Thread(target=self.random_move, args=(character, location, waypoints, speed))
            thread.start()
        except Exception as e:
            print(f"线程发生错误: {e}")
            traceback.print_exc()
