"""
UFL 3D机器人模拟器

改进点：
- 完整的3D支持
- 多种势场类型
- 性能指标计算
- 可视化支持
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

# ============================================================
# 势场定义
# ============================================================

@dataclass
class Goal:
    """目标点 - 产生吸引势场"""
    position: np.ndarray
    strength: float = 5.0
    
    def potential(self, x: np.ndarray) -> float:
        """二次吸引势"""
        return self.strength * np.sum((x - self.position) ** 2)
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """势场梯度"""
        return 2 * self.strength * (x - self.position)

@dataclass
class Obstacle:
    """障碍物 - 产生排斥势场"""
    position: np.ndarray
    radius: float = 0.5
    strength: float = 2.0
    
    def potential(self, x: np.ndarray) -> float:
        """排斥势 - 距离越近势能越高"""
        dist = np.linalg.norm(x - self.position)
        if dist < self.radius:
            return float('inf')
        safe_dist = max(dist - self.radius, 0.1)
        return self.strength / safe_dist
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """排斥梯度"""
        diff = x - self.position
        dist = np.linalg.norm(diff)
        if dist < 0.01:
            return np.zeros_like(x)
        safe_dist = max(dist - self.radius, 0.1)
        direction = diff / dist
        magnitude = self.strength / (safe_dist ** 2)
        return -magnitude * direction

@dataclass
class Boundary:
    """边界约束"""
    min_pos: np.ndarray  # [x_min, y_min, z_min]
    max_pos: np.ndarray  # [x_max, y_max, z_max]
    strength: float = 10.0
    
    def potential(self, x: np.ndarray) -> float:
        """边界势能"""
        pot = 0.0
        margin = 0.5
        
        for i in range(len(x)):
            if x[i] < self.min_pos[i] + margin:
                pot += self.strength / max(x[i] - self.min_pos[i], 0.01)
            if x[i] > self.max_pos[i] - margin:
                pot += self.strength / max(self.max_pos[i] - x[i], 0.01)
        
        return pot
    
    def gradient(self, x: np.ndarray) -> np.ndarray:
        """边界梯度"""
        grad = np.zeros_like(x)
        margin = 0.5
        
        for i in range(len(x)):
            if x[i] < self.min_pos[i] + margin:
                grad[i] -= self.strength / max(x[i] - self.min_pos[i], 0.01)**2
            if x[i] > self.max_pos[i] - margin:
                grad[i] += self.strength / max(self.max_pos[i] - x[i], 0.01)**2
        
        return grad

# ============================================================
# 机器人
# ============================================================

@dataclass
class Robot:
    """3D机器人"""
    position: np.ndarray
    velocity: np.ndarray = None
    max_speed: float = 2.0
    
    def __post_init__(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.position)
        self.trajectory = [self.position.copy()]
        self.velocities = [self.velocity.copy()]
    
    def update(self, total_gradient: np.ndarray, dt: float = 0.05, 
               momentum: float = 0.8, lr: float = 0.5):
        """更新机器人状态"""
        # 梯度裁剪
        grad_norm = np.linalg.norm(total_gradient)
        if grad_norm > 10:
            total_gradient = total_gradient * 10 / grad_norm
        
        # 动量更新
        self.velocity = momentum * self.velocity - lr * total_gradient
        
        # 速度限制
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity * self.max_speed / speed
        
        # 位置更新
        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position.copy())
        self.velocities.append(self.velocity.copy())
    
    def get_trajectory_length(self) -> float:
        """计算轨迹总长度"""
        if len(self.trajectory) < 2:
            return 0.0
        
        total_dist = 0.0
        for i in range(len(self.trajectory) - 1):
            total_dist += np.linalg.norm(self.trajectory[i+1] - self.trajectory[i])
        return total_dist
    
    def get_smoothness(self) -> float:
        """计算轨迹平滑度（加速度的平均值）"""
        if len(self.velocities) < 2:
            return 0.0
        
        accelerations = []
        for i in range(len(self.velocities) - 1):
            acc = np.linalg.norm(self.velocities[i+1] - self.velocities[i])
            accelerations.append(acc)
        
        return np.mean(accelerations) if accelerations else 0.0

# ============================================================
# 3D世界模拟器
# ============================================================

class UFL_World_3D:
    """3D UFL世界模拟器"""
    
    def __init__(self, dimension: int = 3):
        self.dimension = dimension
        self.robot = None
        self.goal = None
        self.obstacles: List[Obstacle] = []
        self.boundary = None
    
    def setup_scenario(self, scenario: str = "simple", dimension: int = 3):
        """设置场景"""
        self.dimension = dimension
        
        if scenario == "simple":
            # 简单场景：一个障碍物
            self.robot = Robot(position=np.array([-3.0, -3.0, -3.0][:dimension]))
            self.goal = Goal(position=np.array([3.0, 3.0, 3.0][:dimension]))
            self.obstacles = [
                Obstacle(position=np.array([0.0, 0.0, 0.0][:dimension]), radius=1.0)
            ]
            self.boundary = Boundary(
                min_pos=np.array([-5.0, -5.0, -5.0][:dimension]),
                max_pos=np.array([5.0, 5.0, 5.0][:dimension])
            )
        
        elif scenario == "maze_3d":
            # 3D迷宫场景
            self.robot = Robot(position=np.array([-4.0, -4.0, -4.0][:dimension]))
            self.goal = Goal(position=np.array([4.0, 4.0, 4.0][:dimension]))
            self.obstacles = [
                Obstacle(position=np.array([-2.0, 0.0, 0.0][:dimension]), radius=0.8),
                Obstacle(position=np.array([0.0, 2.0, 0.0][:dimension]), radius=0.8),
                Obstacle(position=np.array([2.0, 0.0, 0.0][:dimension]), radius=0.8),
                Obstacle(position=np.array([0.0, -2.0, 0.0][:dimension]), radius=0.8),
                Obstacle(position=np.array([0.0, 0.0, 2.0][:dimension]), radius=0.8),
            ]
            self.boundary = Boundary(
                min_pos=np.array([-5.0, -5.0, -5.0][:dimension]),
                max_pos=np.array([5.0, 5.0, 5.0][:dimension])
            )
        
        elif scenario == "narrow_passage":
            # 狭窄通道场景
            self.robot = Robot(position=np.array([-4.0, 0.0, 0.0][:dimension]))
            self.goal = Goal(position=np.array([4.0, 0.0, 0.0][:dimension]))
            
            # 创建狭窄通道
            self.obstacles = [
                Obstacle(position=np.array([-1.0, 1.5, 0.0][:dimension]), radius=0.6),
                Obstacle(position=np.array([-1.0, -1.5, 0.0][:dimension]), radius=0.6),
                Obstacle(position=np.array([1.0, 1.5, 0.0][:dimension]), radius=0.6),
                Obstacle(position=np.array([1.0, -1.5, 0.0][:dimension]), radius=0.6),
            ]
            self.boundary = Boundary(
                min_pos=np.array([-5.0, -5.0, -5.0][:dimension]),
                max_pos=np.array([5.0, 5.0, 5.0][:dimension])
            )
        
        elif scenario == "cluttered":
            # 拥挤场景
            self.robot = Robot(position=np.array([-4.0, -4.0, -4.0][:dimension]))
            self.goal = Goal(position=np.array([4.0, 4.0, 4.0][:dimension]))
            
            # 随机放置多个障碍物
            np.random.seed(42)
            self.obstacles = []
            for _ in range(10):
                pos = np.random.uniform(-3, 3, dimension)
                self.obstacles.append(Obstacle(position=pos, radius=0.5))
            
            self.boundary = Boundary(
                min_pos=np.array([-5.0, -5.0, -5.0][:dimension]),
                max_pos=np.array([5.0, 5.0, 5.0][:dimension])
            )
    
    def total_gradient(self, x: np.ndarray) -> np.ndarray:
        """计算总势场梯度"""
        grad = np.zeros_like(x)
        
        # 目标吸引
        if self.goal:
            grad += self.goal.gradient(x)
        
        # 障碍物排斥
        for obs in self.obstacles:
            grad += obs.gradient(x)
        
        # 边界排斥
        if self.boundary:
            grad += self.boundary.gradient(x)
        
        return grad
    
    def total_potential(self, x: np.ndarray) -> float:
        """计算总势能"""
        pot = 0.0
        
        if self.goal:
            pot += self.goal.potential(x)
        
        for obs in self.obstacles:
            pot += obs.potential(x)
        
        if self.boundary:
            pot += self.boundary.potential(x)
        
        return pot
    
    def step(self, dt: float = 0.05, momentum: float = 0.8, lr: float = 0.5):
        """单步模拟"""
        grad = self.total_gradient(self.robot.position)
        self.robot.update(grad, dt, momentum, lr)
    
    def run(self, max_steps: int = 500, target_dist: float = 0.3, 
            verbose: bool = True) -> Tuple[bool, dict]:
        """运行模拟直到到达目标或超时"""
        
        potentials = []
        distances = []
        
        for step in range(max_steps):
            self.step()
            
            # 记录指标
            pot = self.total_potential(self.robot.position)
            dist_to_goal = np.linalg.norm(self.robot.position - self.goal.position)
            
            potentials.append(pot)
            distances.append(dist_to_goal)
            
            # 检查是否到达目标
            if dist_to_goal < target_dist:
                if verbose:
                    print(f"✓ 到达目标！步数: {step}")
                
                return True, {
                    "success": True,
                    "steps": step,
                    "trajectory_length": self.robot.get_trajectory_length(),
                    "smoothness": self.robot.get_smoothness(),
                    "final_distance": dist_to_goal,
                    "potentials": potentials,
                    "distances": distances
                }
            
            if verbose and step % max(1, max_steps // 10) == 0:
                print(f"步{step:4d} | 距离: {dist_to_goal:.3f} | 势能: {pot:.3f}")
        
        if verbose:
            print(f"✗ 未能到达目标，最终距离: {distances[-1]:.2f}")
        
        return False, {
            "success": False,
            "steps": max_steps,
            "trajectory_length": self.robot.get_trajectory_length(),
            "smoothness": self.robot.get_smoothness(),
            "final_distance": distances[-1],
            "potentials": potentials,
            "distances": distances
        }

# ============================================================
# 性能分析
# ============================================================

@dataclass
class PathMetrics:
    """路径性能指标"""
    success: bool
    steps: int
    trajectory_length: float
    smoothness: float
    final_distance: float
    efficiency: float  # 直线距离 / 实际距离
    
    def __repr__(self):
        status = "✓ 成功" if self.success else "✗ 失败"
        return f"""
路径性能指标
{status}
  步数: {self.steps}
  轨迹长度: {self.trajectory_length:.3f}
  平滑度: {self.smoothness:.3f}
  最终距离: {self.final_distance:.3f}
  效率: {self.efficiency:.2%}
"""

def calculate_metrics(world: UFL_World_3D, result: dict) -> PathMetrics:
    """计算路径性能指标"""
    
    # 直线距离
    start = world.robot.trajectory[0]
    end = world.robot.trajectory[-1]
    goal = world.goal.position
    
    straight_line_dist = np.linalg.norm(goal - start)
    actual_dist = result["trajectory_length"]
    
    efficiency = straight_line_dist / actual_dist if actual_dist > 0 else 0.0
    
    return PathMetrics(
        success=result["success"],
        steps=result["steps"],
        trajectory_length=actual_dist,
        smoothness=result["smoothness"],
        final_distance=result["final_distance"],
        efficiency=efficiency
    )

# ============================================================
# 比较不同场景
# ============================================================

def benchmark_scenarios(dimension: int = 3):
    """对不同场景进行基准测试"""
    
    scenarios = ["simple", "maze_3d", "narrow_passage", "cluttered"]
    results = {}
    
    print(f"\n{'='*60}")
    print(f"3D机器人导航基准测试 (维度: {dimension})")
    print(f"{'='*60}\n")
    
    for scenario in scenarios:
        print(f"场景: {scenario}")
        print("-" * 40)
        
        world = UFL_World_3D(dimension=dimension)
        world.setup_scenario(scenario, dimension=dimension)
        
        success, result = world.run(verbose=False)
        metrics = calculate_metrics(world, result)
        
        print(metrics)
        results[scenario] = metrics
    
    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    
    successful = sum(1 for m in results.values() if m.success)
    print(f"成功率: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    
    avg_efficiency = np.mean([m.efficiency for m in results.values() if m.success])
    print(f"平均效率: {avg_efficiency:.2%}")
    
    return results

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    # 测试2D场景
    print("\n" + "="*60)
    print("2D场景测试")
    print("="*60)
    
    world_2d = UFL_World_3D(dimension=2)
    world_2d.setup_scenario("simple", dimension=2)
    success_2d, result_2d = world_2d.run()
    
    # 测试3D场景
    print("\n" + "="*60)
    print("3D场景测试")
    print("="*60)
    
    world_3d = UFL_World_3D(dimension=3)
    world_3d.setup_scenario("simple", dimension=3)
    success_3d, result_3d = world_3d.run()
    
    # 基准测试
    benchmark_scenarios(dimension=3)
