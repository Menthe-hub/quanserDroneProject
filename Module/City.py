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
from qvl.spline_line import QLabsSplineLine
from qvl.crosswalk import QLabsCrosswalk
from qvl.reference_frame import QLabsReferenceFrame
from qvl.walls import QLabsWalls


def _wait(stop_event, seconds):
    """可被停止事件打断的等待；没传事件时退化为普通 sleep"""
    if stop_event is None:
        time.sleep(seconds)
    else:
        stop_event.wait(seconds)


class City:
    def __init__(self, qlabs):
        self.qlabs=qlabs
        self._movers=[]        # [(线程, 停止事件), ...]，由 periodic_thread / random_thread 登记
    
    def clean_city(self):
        """清空原有建筑物"""
        QLabsBasicShape(self.qlabs).destroy_all_actors_of_class()
        QLabsPerson(self.qlabs).destroy_all_actors_of_class()
        QLabsCharacter(self.qlabs).destroy_all_actors_of_class()
        QLabsTrafficCone(self.qlabs).destroy_all_actors_of_class()
        QLabsStopSign(self.qlabs).destroy_all_actors_of_class()
        
    def clean_all(self):
        """比 clean_city 更彻底：额外清理 spline / crosswalk / reference frame / walls

        clean_city 只管 5 个类（方块、行人、角色、锥桶、停止标志），实测 SplineLine
        和 ReferenceFrame 会残留（见 probe/阶段0-结论.md）。现有两个 Demo 依赖
        clean_city 的原有行为，所以这里另加一个方法而不去改它。

        注意：内部会先调用 stop_pedestrians()，避免销毁 actor 时还有线程在跑。
        """
        self.stop_pedestrians()
        self.clean_city()
        for cls in (QLabsSplineLine, QLabsCrosswalk, QLabsReferenceFrame, QLabsWalls):
            try:
                cls(self.qlabs).destroy_all_actors_of_class()
            except Exception as e:
                print(f"清理 {cls.__name__} 失败: {e}")

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
    
    def periodic_move(self, character, location, waypoints, speed, stop_event=None):
        """折线往复运动
        输入变量：
            character: 行人类
            location: 行人初始位置
            waypoints: 运动过程中所有的途经点
            speed: 移动速度
            stop_event: (可选) threading.Event；被 set 后循环退出。
                        不传则退化为旧版那种停不下来的行为。
        """
        try:
            n=0
            prev_point=location
            while not (stop_event is not None and stop_event.is_set()):
                m=n%len(waypoints)
                character.move_to(waypoints[m], speed)
                _wait(stop_event, math.dist(waypoints[m], prev_point)/speed+1)
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
        stop_event=threading.Event()
        thread=threading.Thread(
            target=self.periodic_move,
            args=(character, location, waypoints, speed, stop_event),
            daemon=True,
            name="City.periodic_move",
        )
        self._movers.append((thread, stop_event))
        thread.start()
        return thread

    def random_move(self, character, location, waypoints, speed, stop_event=None):
        """布朗运动
        输入变量：
            character: 行人类
            location: 行人初始位置
            waypoints: 运动过程中所有的途经点
            speed: 移动速度
            stop_event: (可选) threading.Event；被 set 后循环退出
        """
        try:
            prev_point=location
            while not (stop_event is not None and stop_event.is_set()):
                point=random.sample(waypoints,1)[0]
                character.move_to(point, speed)
                _wait(stop_event, math.dist(point, prev_point)/speed+1)
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
        stop_event=threading.Event()
        thread=threading.Thread(
            target=self.random_move,
            args=(character, location, waypoints, speed, stop_event),
            daemon=True,
            name="City.random_move",
        )
        self._movers.append((thread, stop_event))
        thread.start()
        return thread

    def stop_pedestrians(self, timeout=5.0):
        """停止并回收所有由 *_thread 启动的行人运动线程

        必须在销毁 actor（clean_city / clean_all / destroy_all_actors_of_class）
        之前调用，否则线程会对着已销毁的 actor 调 move_to。

        这是尽力而为的：如果某个行人正卡在 move_to 的中途，最多要等它走完这一段。
        线程本身是 daemon，所以即使没停干净，进程也能正常退出。
        """
        for thread, stop_event in self._movers:
            stop_event.set()
        for thread, stop_event in self._movers:
            thread.join(timeout=timeout)
        self._movers=[]

    # ---------------- 装饰组件 ----------------
    # 碰撞高度规则（实测，见 probe/阶段0-结论.md）：
    #   无人机飞在 z=1.0；道具高于约 0.6 会挡路，矮于 0.6 不挡。
    #   => 矮装饰（路面、车道线、人行横道）可以铺在走廊上；
    #      高装饰（树、路牌、行人）必须放在走廊之外。
    # 另外实测：交通锥、停止标志、SplineLine、Crosswalk、ReferenceFrame 默认都不挡路。

    def create_tree(self, location, scale=1.0, trunk_color=[0.35,0.20,0.10], leaf_color=[0.10,0.50,0.15]):
        """一棵树：圆柱树干 + 圆锥树冠。location 是落地位置（z 取地面高度）。

        **整树必然高于 0.6，只能放在走廊之外。**
        """
        x, y, z=location[0], location[1], location[2]
        trunk_h=1.2*scale
        self.create_building([x, y, z+trunk_h/2], [0,0,0],
                             [0.25*scale, 0.25*scale, trunk_h], trunk_color,
                             configuration=QLabsBasicShape.SHAPE_CYLINDER)
        canopy_h=1.6*scale
        self.create_building([x, y, z+trunk_h+canopy_h/2-0.3*scale], [0,0,0],
                             [1.3*scale, 1.3*scale, canopy_h], leaf_color,
                             configuration=QLabsBasicShape.SHAPE_CONE)

    def create_road(self, points, width=0.6, color=[0.15,0.15,0.15], z=0.02):
        """一条路面（SplineLine 折线）。矮，不挡路，可以铺在走廊上。

        输入变量：
            points: 折线顶点 [[x, y], ...]
            width: 路宽（米）
            color: 颜色
            z: 离地高度
        """
        spline=QLabsSplineLine(self.qlabs)
        spline.spawn(location=[0,0,0], rotation=[0,0,0], scale=[1,1,1])
        spline.set_points(color, [[p[0], p[1], z, width] for p in points])
        return spline

    def create_lane_line(self, center, radius, line_width=0.08, color=[1.0,0.85,0.0], z=0.03):
        """环形车道线（SplineLine 圆）。矮，不挡路。"""
        spline=QLabsSplineLine(self.qlabs)
        spline.spawn(location=[center[0], center[1], z], rotation=[0,0,0], scale=[1,1,1])
        spline.circle_from_center(radius=radius, lineWidth=line_width, color=color)
        return spline

    def create_crosswalk(self, location, rotation=[0,0,0], scale=[1,1,1], configuration=0):
        """人行横道。矮，不挡路。

        注意 QLabsCrosswalk 内部会把 scale 换成 [z, y, x]（spawn 和 spawn_id 各换一次、
        各自只换一次），这里先换回来，保证调用方始终按 [x, y, z] 理解。
        """
        cw=QLabsCrosswalk(self.qlabs)
        cw.spawn(location=location, rotation=rotation,
                 scale=[scale[2], scale[1], scale[0]], configuration=configuration)
        return cw

    def create_reference_frame(self, location, rotation=[0,0,0], scale=[1,1,1]):
        """参考坐标系标记。场景里没有文字标签，用它当标记。不挡路。"""
        rf=QLabsReferenceFrame(self.qlabs)
        rf.spawn(location=location, rotation=rotation, scale=scale)
        return rf

    def create_wall_block(self, location, rotation=[0,0,0], scale=[1,1,3]):
        """泡沫板墙（QLabsWalls）。

        实测（probe/verify_city.py blocking）：**它不挡路** —— 默认不碰撞，
        而且 set_enable_collisions(True) 返回 False（调用失败），
        换大尺寸也一样。所以这是**纯视觉**组件，适合做围挡/装饰；
        **要让某物真正挡住无人机，请用 create_building**（它默认就碰撞）。
        """
        w=QLabsWalls(self.qlabs)
        w.spawn(location=location, rotation=rotation, scale=scale)
        return w
