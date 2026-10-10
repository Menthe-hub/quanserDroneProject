import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import time

class Evaluator:
    def __init__(self):
        self.start_time=None
        self.end_time=None
        self.desired_waypoints=[]
        self.actual_waypoints=[]
        self.waypoints_error=[]
        self.results=None
        
    def start_timer(self):
        """开始计时"""
        self.start_time=time.time()
        
    def stop_timer(self):
        """停止计时"""
        self.end_time=time.time()
        
    def get_flight_duration(self):
        """获取飞行时长"""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time-self.start_time
    
    def add_waypoint(self, desired_point, actual_point):
        """添加轨迹点"""
        self.desired_waypoints.append(desired_point)
        self.actual_waypoints.append(actual_point)

    def evaluate_drone_performance(self, timestamps=None):
        """评估无人机控制性能
        输入变量:
            timestamps: 时间戳数组
        """
        # 转换为numpy数组以确保计算效率
        ideal_waypoints = np.array(self.desired_waypoints)
        actual_waypoints = np.array(self.actual_waypoints)
    
        # 基本检查
        assert ideal_waypoints.shape == actual_waypoints.shape,"理想轨迹点与实际轨迹点形状不一致"
        assert ideal_waypoints.shape[0] > 0, "坐标点不能为空"
    
        n_points = ideal_waypoints.shape[0]
        dimension = ideal_waypoints.shape[1]

        self.waypoints_error = np.sqrt(np.sum((actual_waypoints - ideal_waypoints)**2, axis=1))

        mean_error = np.mean(self.waypoints_error)
        rmse = np.sqrt(np.mean(self.waypoints_error**2))
        max_error = np.max(self.waypoints_error)
        std_error = np.std(self.waypoints_error)
    
        error_50 = np.percentile(self.waypoints_error, 50)  # 中位数误差
        error_95 = np.percentile(self.waypoints_error, 95)  # 95%分位数误差
        error_99 = np.percentile(self.waypoints_error, 99)  # 99%分位数误差
    
        if n_points > 3:
            _, p_value_normal = stats.shapiro(self.waypoints_error)
            is_normal_distribution = p_value_normal > 0.05
        else:
            p_value_normal = is_normal_distribution = None
    
        results = {
            '基本统计': {
                '数据点数': n_points,
                '维度': dimension,
                '平均误差': mean_error,
                '均方根误差(RMSE)': rmse,
                '最大误差': max_error,
                '误差标准差': std_error,
            },
            '百分位数误差': {
                '中位数误差(50%)': error_50,
                '95%分位数误差': error_95,
                '99%分位数误差': error_99,
            },
            '统计检验': {
                '正态分布检验p值': p_value_normal,
                '是否正态分布': is_normal_distribution,
            } if p_value_normal is not None else {'备注': '数据点不足于统计检验'},
            '原始数据': {
                '误差列表': self.waypoints_error,
                '时间戳': timestamps,
            }
        }
        self.results = results
        return results

    def plot_performance_analysis(self, timestamps=None):
        """绘制性能分析图表"""
        if self.results is None:
            raise ValueError("请先生成评估结果")
        errors = self.results['原始数据']['误差列表']
    
        _, axes = plt.subplots(2, 2, figsize=(15, 12))
    
        # 1. 轨迹对比图
        if np.array(self.desired_waypoints).shape[1] >= 2:
            ax1 = axes[0, 0]
            ax1.plot(np.array(self.desired_waypoints)[:, 0], np.array(self.desired_waypoints)[:, 1], 'g-', label='Desired Trajectory', linewidth=2)
            ax1.plot(np.array(self.actual_waypoints)[:, 0], np.array(self.actual_waypoints)[:, 1], 'r-', label='Actual Trajectory', alpha=0.7)
            ax1.scatter(np.array(self.desired_waypoints)[:, 0], np.array(self.desired_waypoints)[:, 1], c='green', s=30, alpha=0.5)
            ax1.scatter(np.array(self.actual_waypoints)[:, 0], np.array(self.actual_waypoints)[:, 1], c='red', s=20, alpha=0.5)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('Trajectory Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
    
        # 2. 误差时间序列
        ax2 = axes[0, 1]
        time_axis = timestamps if timestamps is not None else np.arange(len(errors))
        ax2.plot(time_axis, errors, 'b-', linewidth=1)
        ax2.axhline(y=self.results['基本统计']['平均误差'], color='r', linestyle='--', 
                    label=f"Mean Error: {self.results['基本统计']['平均误差']:.3f}")
        ax2.axhline(y=self.results['基本统计']['均方根误差(RMSE)'], color='orange', linestyle='--', 
                    label=f"RMSE: {self.results['基本统计']['均方根误差(RMSE)']:.3f}")
        ax2.set_xlabel('Time/Sample')
        ax2.set_ylabel('Error')
        ax2.set_title('Error Time Series')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 误差分布直方图
        ax3 = axes[1, 0]
        ax3.hist(errors, bins=min(30, len(errors)//5), alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(self.results['基本统计']['平均误差'], color='red', linestyle='--', 
                    label=f'Mean Error: {self.results["基本统计"]["平均误差"]:.3f}')
        ax3.axvline(self.results['百分位数误差']['中位数误差(50%)'], color='green', linestyle='--', 
                    label=f'Median: {self.results["百分位数误差"]["中位数误差(50%)"]:.3f}')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Error Distribution Histogram')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 箱线图显示统计分布
        ax4 = axes[1, 1]
        ax4.boxplot(errors, vert=True)
        ax4.set_ylabel('Error')
        ax4.set_title('Error Distribution Boxplot')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_evaluation_report(self, finished, timestamps=None):
        """生成完整的评估报告
        输入变量：
            finished: 无人机是否成功完成任务
            timestamps: 时间戳数组
        """
        if self.results is None:
            self.evaluate_drone_performance(timestamps)
    
        duration = self.get_flight_duration()
    
        # 获取性能评级
        performance_rating = self._get_performance_rating(duration, finished)
    
        # 生成详细报告
        report = f"""
        {'='*60}
        无人机飞行性能评估报告
        {'='*60}

        飞行基本信息:
            飞行时长: {duration:.2f} 秒

        轨迹精度评估:
            平均误差: {self.results['基本统计']['平均误差']:.6f}
            均方根误差(RMSE): {self.results['基本统计']['均方根误差(RMSE)']:.6f}
            最大误差: {self.results['基本统计']['最大误差']:.6f}
            误差标准差: {self.results['基本统计']['误差标准差']:.6f}

        误差分布统计:
            中位数误差(50%): {self.results['百分位数误差']['中位数误差(50%)']:.6f}
            95%分位数误差: {self.results['百分位数误差']['95%分位数误差']:.6f}
            99%分位数误差: {self.results['百分位数误差']['99%分位数误差']:.6f}

        统计检验结果:
        """
    
        # 添加统计检验结果
        if '统计检验' in self.results and self.results['统计检验'].get('正态分布检验p值') is not None:
            normal_test = self.results['统计检验']
            report += f"    正态分布检验p值: {normal_test['正态分布检验p值']:.6f}\n"
            report += f"    误差分布是否正态: {'是' if normal_test['是否正态分布'] else '否'}\n"
        else:
            report += "    统计检验: 数据点不足，无法进行正态性检验\n"
    
        # 性能评级部分
        report += f"""
        性能综合评级:
            评分: {performance_rating:.2f}/100
        """
        report += f"\n{'='*60}"

        return report

    def _get_performance_rating(self, duration, finished):
        """计算综合性能评分"""
        if self.results is None or len(self.waypoints_error)==0 or not finished:
            return 0

        # 获取关键指标
        mean_error = self.results['基本统计']['平均误差']
        max_error = self.results['基本统计']['最大误差']
        rmse = self.results['基本统计']['均方根误差(RMSE)']

        # 每0.1误差扣5分
        error_score = max(0, 100 - (mean_error * 50))

        # RMSE惩罚项
        rmse_penalty = max(0, (rmse - mean_error) * 20)

        # 最大误差惩罚项
        max_error_penalty = max(0, (max_error - mean_error * 3) * 5)

        # 飞行时长稳定性奖励
        duration_bonus = 0
        if duration > 0:
            if duration <= 70:
                duration_bonus = 100-(duration-50)*2
            else:
                duration_bonus = 60-(duration-70)*7

        # 计算总分
        total_score = max(0, min(100, error_score*0.7+duration_bonus*0.3-rmse_penalty-max_error_penalty))
        
        return total_score