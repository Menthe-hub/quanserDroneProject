import time

class PIDController:
    def __init__(self, kp, ki, kd, setpoint, output_limits=(-5.0, 5.0)):
        """
        输入变量：
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            setpoint: 目标值
            output_limits: 输出限制
        """
        self.kp=kp
        self.ki=ki
        self.kd=kd
        self.setpoint=setpoint
        self.output_limits=output_limits
        
        self.integral=0
        self.prev_error=0
        self.prev_time=time.time()
    
    def reset(self):
        """重置误差项和时间"""
        self.prev_error=0
        self.integral=0
        self.prev_time=time.time()
    
    def update(self, measurement, dt=0.1):
        """计算输出指令
        输入变量：
            measurement: 测量值
            dt: 差分时间
        """
        current_time=time.time()
        if dt<=0:
            return 0
        
        error=self.setpoint-measurement
        
        p_term=self.kp*error
        
        self.integral+=error*dt
        i_term=self.ki*self.integral
        
        derivative = (error-self.prev_error) / dt
        d_term=self.kd*derivative
        
        output=max(self.output_limits[0], min(p_term+i_term+d_term, self.output_limits[1]))
        
        self.prev_error=error
        self.prev_time=current_time
        
        return output