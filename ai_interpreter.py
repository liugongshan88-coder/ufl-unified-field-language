"""
AI黑盒解释工具

使用UFL框架来解释神经网络的内部计算过程。

核心思想：
1. 将神经网络的隐层激活看作动力系统的轨迹
2. 尝试恢复驱动这个轨迹的势函数
3. 可视化和分析这个势函数
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
import warnings

# ============================================================
# 神经网络轨迹提取
# ============================================================

@dataclass
class NetworkTrajectory:
    """神经网络的轨迹数据"""
    layer_outputs: List[np.ndarray]  # 每层的激活
    layer_names: List[str]
    input_data: np.ndarray
    output_data: np.ndarray
    
    def get_dimension(self) -> int:
        """获取隐层维度"""
        return self.layer_outputs[0].shape[1] if len(self.layer_outputs) > 0 else 0
    
    def get_num_samples(self) -> int:
        """获取样本数"""
        return self.layer_outputs[0].shape[0] if len(self.layer_outputs) > 0 else 0
    
    def get_trajectory_for_sample(self, sample_idx: int) -> np.ndarray:
        """获取单个样本的轨迹"""
        trajectory = []
        for layer_output in self.layer_outputs:
            trajectory.append(layer_output[sample_idx])
        return np.array(trajectory)

class NeuralNetworkInterpreter:
    """神经网络解释器"""
    
    def __init__(self, trajectory: NetworkTrajectory):
        self.trajectory = trajectory
        self.potential_function = None
        self.gradient_function = None
    
    def extract_trajectories(self) -> List[np.ndarray]:
        """提取所有样本的轨迹"""
        trajectories = []
        for i in range(self.trajectory.get_num_samples()):
            traj = self.trajectory.get_trajectory_for_sample(i)
            trajectories.append(traj)
        return trajectories
    
    def estimate_velocities(self, trajectories: List[np.ndarray]) -> List[np.ndarray]:
        """估计轨迹的速度（层间差分）"""
        velocities = []
        for traj in trajectories:
            vel = []
            for i in range(len(traj) - 1):
                v = traj[i+1] - traj[i]
                vel.append(v)
            velocities.append(np.array(vel))
        return velocities
    
    def estimate_potential_gradient(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计单个样本的势函数梯度
        
        假设：dx/dt ≈ -∇V(x)
        
        所以：∇V(x) ≈ -dx/dt
        """
        traj = self.trajectory.get_trajectory_for_sample(sample_idx)
        
        # 计算速度（层间差分）
        velocities = []
        for i in range(len(traj) - 1):
            v = traj[i+1] - traj[i]
            velocities.append(v)
        
        # 梯度估计：-v
        gradients = [-v for v in velocities]
        
        return traj, np.array(gradients)
    
    def fit_potential_function(self, degree: int = 2) -> Callable:
        """
        拟合势函数
        
        使用多项式拟合：V(x) ≈ sum(a_i * x_i^2) + sum(b_ij * x_i * x_j) + ...
        """
        
        # 收集所有的 (x, -dx/dt) 对
        X_data = []
        grad_data = []
        
        for i in range(self.trajectory.get_num_samples()):
            traj, grads = self.estimate_potential_gradient(i)
            
            # 使用轨迹上的所有点
            for j in range(len(traj) - 1):
                X_data.append(traj[j])
                grad_data.append(grads[j])
        
        X_data = np.array(X_data)
        grad_data = np.array(grad_data)
        
        # 二次势函数：V(x) = 0.5 * x^T * A * x + b^T * x + c
        # 梯度：∇V(x) = A * x + b
        
        # 使用最小二乘法拟合 A 和 b
        # grad_data ≈ A * X_data + b
        
        # 添加偏置项
        X_augmented = np.column_stack([X_data, np.ones(len(X_data))])
        
        # 最小二乘求解
        try:
            # 对每个梯度分量分别拟合
            A_list = []
            b_list = []
            
            for d in range(grad_data.shape[1]):
                coeffs = np.linalg.lstsq(X_augmented, grad_data[:, d], rcond=None)[0]
                A_list.append(coeffs[:-1])
                b_list.append(coeffs[-1])
            
            A = np.array(A_list)
            b = np.array(b_list)
            
            # 创建势函数
            def potential(x):
                return 0.5 * np.dot(x, np.dot(A, x)) + np.dot(b, x)
            
            def gradient(x):
                return np.dot(A, x) + b
            
            self.potential_function = potential
            self.gradient_function = gradient
            
            return potential
        
        except Exception as e:
            warnings.warn(f"势函数拟合失败: {e}")
            return None
    
    def analyze_critical_points(self) -> Dict:
        """分析临界点（梯度为零的点）"""
        
        if self.gradient_function is None:
            self.fit_potential_function()
        
        if self.gradient_function is None:
            return {}
        
        # 收集所有轨迹点
        all_points = []
        for i in range(self.trajectory.get_num_samples()):
            traj = self.trajectory.get_trajectory_for_sample(i)
            all_points.extend(traj)
        
        all_points = np.array(all_points)
        
        # 计算所有点的梯度范数
        gradient_norms = []
        for point in all_points:
            grad = self.gradient_function(point)
            gradient_norms.append(np.linalg.norm(grad))
        
        gradient_norms = np.array(gradient_norms)
        
        # 找到梯度最小的点
        min_idx = np.argmin(gradient_norms)
        critical_point = all_points[min_idx]
        
        return {
            "critical_point": critical_point,
            "gradient_norm": gradient_norms[min_idx],
            "mean_gradient_norm": np.mean(gradient_norms),
            "max_gradient_norm": np.max(gradient_norms),
            "min_gradient_norm": np.min(gradient_norms)
        }
    
    def analyze_convergence(self) -> Dict:
        """分析收敛性"""
        
        # 计算每个样本的轨迹长度
        trajectory_lengths = []
        
        for i in range(self.trajectory.get_num_samples()):
            traj = self.trajectory.get_trajectory_for_sample(i)
            length = 0.0
            for j in range(len(traj) - 1):
                length += np.linalg.norm(traj[j+1] - traj[j])
            trajectory_lengths.append(length)
        
        trajectory_lengths = np.array(trajectory_lengths)
        
        return {
            "mean_trajectory_length": np.mean(trajectory_lengths),
            "std_trajectory_length": np.std(trajectory_lengths),
            "max_trajectory_length": np.max(trajectory_lengths),
            "min_trajectory_length": np.min(trajectory_lengths)
        }
    
    def analyze_stability(self) -> Dict:
        """分析稳定性"""
        
        if self.gradient_function is None:
            self.fit_potential_function()
        
        if self.gradient_function is None:
            return {}
        
        # 计算Hessian矩阵的特征值（数值估计）
        dim = self.trajectory.get_dimension()
        
        # 在临界点处估计Hessian
        critical_info = self.analyze_critical_points()
        x_critical = critical_info["critical_point"]
        
        # 数值Hessian
        eps = 1e-5
        H = np.zeros((dim, dim))
        
        for i in range(dim):
            for j in range(dim):
                x_pp = x_critical.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                
                x_pm = x_critical.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                
                x_mp = x_critical.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                
                x_mm = x_critical.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                
                grad_pp = self.gradient_function(x_pp)
                grad_pm = self.gradient_function(x_pm)
                grad_mp = self.gradient_function(x_mp)
                grad_mm = self.gradient_function(x_mm)
                
                H[i, j] = (grad_pp[i] - grad_pm[i] - grad_mp[i] + grad_mm[i]) / (4 * eps * eps)
        
        # 计算特征值
        eigenvalues = np.linalg.eigvals(H)
        
        return {
            "eigenvalues": eigenvalues,
            "all_positive": np.all(eigenvalues > 0),
            "all_negative": np.all(eigenvalues < 0),
            "mixed_signs": not (np.all(eigenvalues > 0) or np.all(eigenvalues < 0)),
            "condition_number": np.max(np.abs(eigenvalues)) / (np.min(np.abs(eigenvalues)) + 1e-10)
        }
    
    def generate_report(self) -> str:
        """生成完整的分析报告"""
        
        report = []
        report.append("=" * 60)
        report.append("神经网络黑盒解释报告")
        report.append("=" * 60)
        
        # 基本信息
        report.append("\n基本信息：")
        report.append(f"  隐层维度: {self.trajectory.get_dimension()}")
        report.append(f"  样本数: {self.trajectory.get_num_samples()}")
        report.append(f"  层数: {len(self.trajectory.layer_names)}")
        
        # 拟合势函数
        report.append("\n拟合势函数...")
        self.fit_potential_function()
        report.append("  ✓ 完成")
        
        # 临界点分析
        report.append("\n临界点分析：")
        critical_info = self.analyze_critical_points()
        report.append(f"  梯度范数: {critical_info['gradient_norm']:.6f}")
        report.append(f"  平均梯度范数: {critical_info['mean_gradient_norm']:.6f}")
        report.append(f"  最大梯度范数: {critical_info['max_gradient_norm']:.6f}")
        
        # 收敛性分析
        report.append("\n收敛性分析：")
        conv_info = self.analyze_convergence()
        report.append(f"  平均轨迹长度: {conv_info['mean_trajectory_length']:.3f}")
        report.append(f"  标准差: {conv_info['std_trajectory_length']:.3f}")
        
        # 稳定性分析
        report.append("\n稳定性分析：")
        stab_info = self.analyze_stability()
        if "eigenvalues" in stab_info:
            eigenvalues = stab_info["eigenvalues"]
            report.append(f"  特征值: {eigenvalues}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)

# ============================================================
# 示例：简单神经网络
# ============================================================

def create_simple_network_trajectory() -> NetworkTrajectory:
    """创建一个简单的神经网络轨迹示例"""
    
    np.random.seed(42)
    
    # 模拟神经网络的层输出
    num_samples = 100
    num_layers = 4
    hidden_dim = 10
    
    # 输入
    X = np.random.randn(num_samples, 5)
    
    # 模拟层输出（每层都是前一层的非线性变换）
    layer_outputs = []
    h = X
    
    for layer_idx in range(num_layers):
        # 随机权重
        W = np.random.randn(h.shape[1], hidden_dim) * 0.1
        b = np.random.randn(hidden_dim) * 0.01
        
        # 前向传播
        h = np.tanh(np.dot(h, W) + b)
        layer_outputs.append(h)
    
    # 输出
    W_out = np.random.randn(hidden_dim, 1) * 0.1
    y = np.dot(h, W_out)
    
    return NetworkTrajectory(
        layer_outputs=layer_outputs,
        layer_names=[f"Layer {i}" for i in range(num_layers)],
        input_data=X,
        output_data=y
    )

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("AI黑盒解释工具演示")
    print("=" * 60)
    
    # 创建示例轨迹
    print("\n创建神经网络轨迹...")
    trajectory = create_simple_network_trajectory()
    print(f"  维度: {trajectory.get_dimension()}")
    print(f"  样本数: {trajectory.get_num_samples()}")
    print(f"  层数: {len(trajectory.layer_names)}")
    
    # 创建解释器
    print("\n初始化解释器...")
    interpreter = NeuralNetworkInterpreter(trajectory)
    
    # 生成报告
    print("\n生成分析报告...\n")
    report = interpreter.generate_report()
    print(report)
    
    # 详细分析
    print("\n详细分析：")
    
    print("\n1. 临界点分析")
    critical = interpreter.analyze_critical_points()
    print(f"   梯度范数: {critical['gradient_norm']:.6f}")
    
    print("\n2. 收敛性分析")
    convergence = interpreter.analyze_convergence()
    print(f"   平均轨迹长度: {convergence['mean_trajectory_length']:.3f}")
    
    print("\n3. 稳定性分析")
    stability = interpreter.analyze_stability()
    if "eigenvalues" in stability:
        print(f"   特征值: {stability['eigenvalues']}")
        print(f"   条件数: {stability['condition_number']:.3f}")
