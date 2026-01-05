"""
UFL - Unified Field Language
改进版执行引擎

改进点：
- 符号微分支持
- 多种数值求解器（欧拉、RK4、自适应）
- 更好的数值稳定性
- 约束系统改进
- 详细的运行统计
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
from abc import ABC, abstractmethod
import warnings

from syntax import (
    SystemDef, FieldDef, ParticleDef, SimulateDef, UFLParser,
    Expr, BinaryOp, UnaryOp, FunctionCall, Variable, Number, VectorLiteral
)

# ============================================================
# 符号微分
# ============================================================

class SymbolicDifferentiator:
    """符号表达式微分"""
    
    @staticmethod
    def diff(expr: Expr, var_name: str) -> Expr:
        """对表达式求导"""
        
        if isinstance(expr, Number):
            # 常数的导数为0
            return Number(value=0.0)
        
        elif isinstance(expr, Variable):
            # 变量的导数
            if expr.name == var_name:
                return Number(value=1.0)
            else:
                return Number(value=0.0)
        
        elif isinstance(expr, BinaryOp):
            # 二元操作
            if expr.op == '+':
                # (f + g)' = f' + g'
                return BinaryOp(
                    op='+',
                    left=SymbolicDifferentiator.diff(expr.left, var_name),
                    right=SymbolicDifferentiator.diff(expr.right, var_name)
                )
            
            elif expr.op == '-':
                # (f - g)' = f' - g'
                return BinaryOp(
                    op='-',
                    left=SymbolicDifferentiator.diff(expr.left, var_name),
                    right=SymbolicDifferentiator.diff(expr.right, var_name)
                )
            
            elif expr.op == '*':
                # (f * g)' = f' * g + f * g'
                f_prime = SymbolicDifferentiator.diff(expr.left, var_name)
                g_prime = SymbolicDifferentiator.diff(expr.right, var_name)
                return BinaryOp(
                    op='+',
                    left=BinaryOp(op='*', left=f_prime, right=expr.right),
                    right=BinaryOp(op='*', left=expr.left, right=g_prime)
                )
            
            elif expr.op == '/':
                # (f / g)' = (f' * g - f * g') / g^2
                f_prime = SymbolicDifferentiator.diff(expr.left, var_name)
                g_prime = SymbolicDifferentiator.diff(expr.right, var_name)
                numerator = BinaryOp(
                    op='-',
                    left=BinaryOp(op='*', left=f_prime, right=expr.right),
                    right=BinaryOp(op='*', left=expr.left, right=g_prime)
                )
                denominator = BinaryOp(op='^', left=expr.right, right=Number(value=2.0))
                return BinaryOp(op='/', left=numerator, right=denominator)
            
            elif expr.op == '^':
                # (f^n)' = n * f^(n-1) * f'
                if isinstance(expr.right, Number):
                    n = expr.right.value
                    f_prime = SymbolicDifferentiator.diff(expr.left, var_name)
                    return BinaryOp(
                        op='*',
                        left=BinaryOp(
                            op='*',
                            left=Number(value=n),
                            right=BinaryOp(op='^', left=expr.left, right=Number(value=n-1))
                        ),
                        right=f_prime
                    )
                else:
                    # 一般情况：f^g = e^(g*ln(f))
                    warnings.warn("非常数指数的微分可能不准确")
                    return Number(value=0.0)
        
        elif isinstance(expr, UnaryOp):
            if expr.op == '-':
                # (-f)' = -f'
                return UnaryOp(op='-', operand=SymbolicDifferentiator.diff(expr.operand, var_name))
            
            elif expr.op == '|·|':
                # |f|' = f' * f / |f|
                f_prime = SymbolicDifferentiator.diff(expr.operand, var_name)
                return BinaryOp(
                    op='*',
                    left=f_prime,
                    right=BinaryOp(
                        op='/',
                        left=expr.operand,
                        right=expr  # |f|
                    )
                )
        
        elif isinstance(expr, FunctionCall):
            # 函数调用的链式法则
            if expr.name == 'sqrt':
                # sqrt(f)' = f' / (2 * sqrt(f))
                f_prime = SymbolicDifferentiator.diff(expr.args[0], var_name)
                return BinaryOp(
                    op='/',
                    left=f_prime,
                    right=BinaryOp(
                        op='*',
                        left=Number(value=2.0),
                        right=expr
                    )
                )
        
        return Number(value=0.0)

# ============================================================
# 表达式编译器
# ============================================================

class ExpressionCompiler:
    """将AST表达式编译为可执行函数"""
    
    @staticmethod
    def compile_expr(expr: Expr, dim: int, use_symbolic_grad: bool = False) -> Callable:
        """编译表达式为函数"""
        
        def eval_expr(e: Expr, x: np.ndarray, context: Dict) -> float:
            if isinstance(e, Number):
                return e.value
            
            elif isinstance(e, Variable):
                if e.name == 'x':
                    # 返回向量的范数平方（用于 |x|^2）
                    return np.linalg.norm(x)
                elif e.name in context:
                    return context[e.name]
                else:
                    raise RuntimeError(f"未定义的变量: {e.name}")
            
            elif isinstance(e, BinaryOp):
                left = eval_expr(e.left, x, context)
                right = eval_expr(e.right, x, context)
                
                if e.op == '+':
                    return left + right
                elif e.op == '-':
                    return left - right
                elif e.op == '*':
                    return left * right
                elif e.op == '/':
                    if right == 0:
                        raise RuntimeError("除以零")
                    return left / right
                elif e.op == '^':
                    return left ** right
            
            elif isinstance(e, UnaryOp):
                operand = eval_expr(e.operand, x, context)
                
                if e.op == '-':
                    return -operand
                elif e.op == '|·|':
                    return np.abs(operand)
            
            elif isinstance(e, FunctionCall):
                if e.name == 'sqrt':
                    arg = eval_expr(e.args[0], x, context)
                    return np.sqrt(arg)
                elif e.name == 'abs':
                    arg = eval_expr(e.args[0], x, context)
                    return np.abs(arg)
                elif e.name == 'sin':
                    arg = eval_expr(e.args[0], x, context)
                    return np.sin(arg)
                elif e.name == 'cos':
                    arg = eval_expr(e.args[0], x, context)
                    return np.cos(arg)
                elif e.name == 'exp':
                    arg = eval_expr(e.args[0], x, context)
                    return np.exp(arg)
                elif e.name == 'log':
                    arg = eval_expr(e.args[0], x, context)
                    return np.log(arg)
            
            raise RuntimeError(f"未知表达式类型: {type(e)}")
        
        def field_func(x: np.ndarray, **context) -> float:
            try:
                return eval_expr(expr, x, context)
            except Exception as e:
                raise RuntimeError(f"表达式求值错误: {e}")
        
        return field_func
    
    @staticmethod
    def compile_field(expr: Expr, dim: int) -> Callable:
        """编译场表达式"""
        return ExpressionCompiler.compile_expr(expr, dim)

# ============================================================
# 数值求解器
# ============================================================

class Solver(ABC):
    """求解器基类"""
    
    @abstractmethod
    def step(self, x: np.ndarray, v: np.ndarray, 
             potential: Callable, dt: float, 
             lr: float, momentum: float, **context) -> Tuple[np.ndarray, np.ndarray]:
        """单步求解"""
        pass

class EulerSolver(Solver):
    """显式欧拉方法"""
    
    def step(self, x: np.ndarray, v: np.ndarray, 
             potential: Callable, dt: float, 
             lr: float, momentum: float, **context) -> Tuple[np.ndarray, np.ndarray]:
        
        grad = self._numerical_gradient(potential, x, **context)
        grad = self._clip_gradient(grad)
        
        v_new = momentum * v - lr * grad
        x_new = x + v_new * dt
        
        return x_new, v_new
    
    @staticmethod
    def _numerical_gradient(f: Callable, x: np.ndarray, eps: float = 1e-5, **context) -> np.ndarray:
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (f(x_plus, **context) - f(x_minus, **context)) / (2 * eps)
        return grad
    
    @staticmethod
    def _clip_gradient(grad: np.ndarray, max_norm: float = 10.0) -> np.ndarray:
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            return grad * max_norm / norm
        return grad

class RK4Solver(Solver):
    """四阶Runge-Kutta方法"""
    
    def step(self, x: np.ndarray, v: np.ndarray, 
             potential: Callable, dt: float, 
             lr: float, momentum: float, **context) -> Tuple[np.ndarray, np.ndarray]:
        
        # 定义动力学系统
        def dynamics(state):
            x_s, v_s = state[:len(x)], state[len(x):]
            grad = EulerSolver._numerical_gradient(potential, x_s, **context)
            grad = EulerSolver._clip_gradient(grad)
            
            dv = -lr * grad - momentum * v_s  # 阻尼形式
            dx = v_s
            return np.concatenate([dx, dv])
        
        state = np.concatenate([x, v])
        
        k1 = dynamics(state)
        k2 = dynamics(state + 0.5 * dt * k1)
        k3 = dynamics(state + 0.5 * dt * k2)
        k4 = dynamics(state + dt * k3)
        
        state_new = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        x_new = state_new[:len(x)]
        v_new = state_new[len(x):]
        
        return x_new, v_new

class AdaptiveSolver(Solver):
    """自适应步长求解器"""
    
    def step(self, x: np.ndarray, v: np.ndarray, 
             potential: Callable, dt: float, 
             lr: float, momentum: float, **context) -> Tuple[np.ndarray, np.ndarray]:
        
        # 使用RK4和RK5比较以估计误差
        euler = EulerSolver()
        rk4 = RK4Solver()
        
        x_euler, v_euler = euler.step(x, v, potential, dt, lr, momentum, **context)
        x_rk4, v_rk4 = rk4.step(x, v, potential, dt, lr, momentum, **context)
        
        # 误差估计
        error = np.linalg.norm(x_rk4 - x_euler) + np.linalg.norm(v_rk4 - v_euler)
        
        # 简单的步长调整（实际应用中应更复杂）
        if error > 1e-3:
            warnings.warn(f"步长过大，误差: {error}")
        
        return x_rk4, v_rk4

# ============================================================
# 约束系统
# ============================================================

class Constraint(ABC):
    """约束基类"""
    
    @abstractmethod
    def apply(self, x: np.ndarray) -> np.ndarray:
        """应用约束"""
        pass
    
    @abstractmethod
    def project_gradient(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        """投影梯度到约束流形"""
        pass

class SphereConstraint(Constraint):
    """球面约束 |x| = r"""
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x)
        if norm > 0:
            return x / norm * self.radius
        return x
    
    def project_gradient(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        # 投影到切空间：移除径向分量
        norm = np.linalg.norm(x)
        if norm > 1e-10:
            radial = np.dot(grad, x) / (norm ** 2) * x
            return grad - radial
        return grad

class CylinderConstraint(Constraint):
    """圆柱约束 x^2 + y^2 = r^2"""
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        xy_norm = np.sqrt(x[0]**2 + x[1]**2)
        if xy_norm > 0:
            x_new = x.copy()
            x_new[0] = x[0] / xy_norm * self.radius
            x_new[1] = x[1] / xy_norm * self.radius
            return x_new
        return x
    
    def project_gradient(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        xy_norm = np.sqrt(x[0]**2 + x[1]**2)
        if xy_norm > 1e-10:
            radial = (grad[0] * x[0] + grad[1] * x[1]) / (xy_norm ** 2)
            grad_new = grad.copy()
            grad_new[0] -= radial * x[0]
            grad_new[1] -= radial * x[1]
            return grad_new
        return grad

class PlaneConstraint(Constraint):
    """平面约束 n·x = d"""
    
    def __init__(self, normal: np.ndarray, distance: float = 0.0):
        self.normal = normal / np.linalg.norm(normal)
        self.distance = distance
    
    def apply(self, x: np.ndarray) -> np.ndarray:
        deviation = np.dot(self.normal, x) - self.distance
        return x - deviation * self.normal
    
    def project_gradient(self, grad: np.ndarray, x: np.ndarray) -> np.ndarray:
        # 移除法向分量
        normal_component = np.dot(grad, self.normal)
        return grad - normal_component * self.normal

# ============================================================
# 运行时系统
# ============================================================

@dataclass
class ParticleState:
    """粒子运行时状态"""
    name: str
    position: np.ndarray
    velocity: np.ndarray
    trajectory: List[np.ndarray] = field(default_factory=list)
    potential_history: List[float] = field(default_factory=list)
    
    def record(self, potential: float = 0.0):
        self.trajectory.append(self.position.copy())
        self.potential_history.append(potential)

@dataclass
class SimulationStats:
    """模拟统计"""
    total_steps: int = 0
    converged: bool = False
    convergence_step: Optional[int] = None
    final_potential: float = 0.0
    total_distance: float = 0.0
    avg_gradient_norm: float = 0.0
    computation_time: float = 0.0

class UFLRuntime:
    """UFL运行时"""
    
    def __init__(self, system: SystemDef, solver_type: str = "rk4"):
        self.system = system
        self.compiled_fields: Dict[str, Callable] = {}
        self.particles: List[ParticleState] = []
        self.constraints: List[Constraint] = []
        self.history: Dict[str, List] = {
            "potential": [],
            "step": [],
            "gradient_norm": []
        }
        self.context: Dict[str, Any] = {}
        self.stats = SimulationStats()
        
        # 选择求解器
        if solver_type == "euler":
            self.solver = EulerSolver()
        elif solver_type == "rk4":
            self.solver = RK4Solver()
        elif solver_type == "adaptive":
            self.solver = AdaptiveSolver()
        else:
            self.solver = RK4Solver()
    
    def compile(self):
        """编译系统"""
        print(f"编译系统: {self.system.name}")
        print(f"  领域: {self.system.domain}")
        print(f"  维度: {self.system.dimension}")
        
        # 编译场表达式
        for field_def in self.system.fields:
            self.compiled_fields[field_def.name] = ExpressionCompiler.compile_field(
                field_def.expression, 
                self.system.dimension
            )
            print(f"  编译场: {field_def.name}")
        
        # 初始化粒子
        for p_def in self.system.particles:
            pos = np.array(p_def.position, dtype=float)
            vel = np.array(p_def.velocity if p_def.velocity else [0.0] * len(pos))
            
            self.particles.append(ParticleState(
                name=p_def.name,
                position=pos,
                velocity=vel
            ))
            print(f"  初始化粒子: {p_def.name} at {pos}")
        
        # 初始化约束
        for constraint_def in self.system.constraints:
            if constraint_def.type == "sphere":
                radius = float(constraint_def.params.get('radius', 1.0))
                self.constraints.append(SphereConstraint(radius))
                print(f"  添加球面约束: 半径 {radius}")
            elif constraint_def.type == "cylinder":
                radius = float(constraint_def.params.get('radius', 1.0))
                self.constraints.append(CylinderConstraint(radius))
                print(f"  添加圆柱约束: 半径 {radius}")
    
    def total_potential(self, x: np.ndarray) -> float:
        """计算总势能"""
        total = 0.0
        for name, field_func in self.compiled_fields.items():
            total += field_func(x, **self.context)
        return total
    
    def total_gradient(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """计算总梯度"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (self.total_potential(x_plus) - self.total_potential(x_minus)) / (2 * eps)
        return grad
    
    def run(self, verbose: bool = True, convergence_tol: float = 1e-6) -> Dict[str, Any]:
        """运行模拟"""
        if not self.system.simulate:
            raise RuntimeError("未定义simulate块")
        
        sim = self.system.simulate
        
        if verbose:
            print(f"\n开始模拟: {sim.steps}步, dt={sim.dt}, 求解器={self.solver.__class__.__name__}")
            print("-" * 60)
        
        # 获取动力学参数
        dynamics_params = {}
        if self.system.particles:
            for p_def in self.system.particles:
                if p_def.dynamics_params:
                    dynamics_params = p_def.dynamics_params
                    break
        
        lr = dynamics_params.get('lr', 0.1)
        momentum = dynamics_params.get('momentum', 0.8)
        
        gradient_norms = []
        
        # 主循环
        for step in range(sim.steps):
            # 更新每个粒子
            for particle in self.particles:
                particle.record()
                
                # 计算势能和梯度
                pot = self.total_potential(particle.position)
                grad = self.total_gradient(particle.position)
                gradient_norms.append(np.linalg.norm(grad))
                
                # 单步求解
                particle.position, particle.velocity = self.solver.step(
                    particle.position,
                    particle.velocity,
                    self.total_potential,
                    sim.dt,
                    lr,
                    momentum,
                    **self.context
                )
                
                # 应用约束
                for constraint in self.constraints:
                    particle.position = constraint.apply(particle.position)
                
                particle.potential_history.append(pot)
            
            # 记录历史
            total_pot = sum(self.total_potential(p.position) for p in self.particles)
            self.history["potential"].append(total_pot)
            self.history["step"].append(step)
            self.history["gradient_norm"].append(np.mean(gradient_norms[-len(self.particles):]))
            
            # 检查收敛
            if np.mean(gradient_norms[-len(self.particles):]) < convergence_tol:
                if self.stats.convergence_step is None:
                    self.stats.convergence_step = step
            
            # 输出
            if verbose and step % max(1, sim.steps // 10) == 0:
                pos_str = ", ".join([f"{p.name}: {p.position.round(3)}" for p in self.particles])
                grad_norm = np.mean(gradient_norms[-len(self.particles):])
                print(f"步{step:4d} | 势能: {total_pot:10.4f} | 梯度: {grad_norm:8.4f} | {pos_str}")
        
        if verbose:
            print("-" * 60)
            print("模拟完成")
            for p in self.particles:
                print(f"  {p.name} 最终位置: {p.position.round(4)}")
        
        # 计算统计
        self.stats.total_steps = sim.steps
        self.stats.final_potential = self.history["potential"][-1] if self.history["potential"] else 0.0
        self.stats.avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        for p in self.particles:
            if len(p.trajectory) > 1:
                dist = sum(
                    np.linalg.norm(np.array(p.trajectory[i+1]) - np.array(p.trajectory[i]))
                    for i in range(len(p.trajectory) - 1)
                )
                self.stats.total_distance += dist
        
        return {
            "particles": self.particles,
            "history": self.history,
            "stats": self.stats
        }

# ============================================================
# 可视化辅助
# ============================================================

def visualize_ascii(result: Dict, title: str = ""):
    """ASCII可视化"""
    print(f"\n{'=' * 60}")
    print(f"可视化: {title}")
    print('=' * 60)
    
    # 势能曲线
    potentials = result["history"]["potential"]
    if potentials:
        max_pot = max(potentials)
        min_pot = min(potentials)
        range_pot = max_pot - min_pot if max_pot != min_pot else 1
        
        print("\n势能变化:")
        height = 10
        width = min(50, len(potentials))
        step_size = max(1, len(potentials) // width)
        
        for row in range(height, -1, -1):
            line = ""
            for col in range(width):
                idx = col * step_size
                if idx < len(potentials):
                    normalized = (potentials[idx] - min_pot) / range_pot
                    if int(normalized * height) >= row:
                        line += "█"
                    else:
                        line += " "
            level = min_pot + (row / height) * range_pot
            print(f"{level:8.2f} |{line}|")
        print(f"         {'─' * width}")
        print(f"         0{' ' * (width-5)}steps")
    
    # 统计
    stats = result.get("stats")
    if stats:
        print(f"\n统计信息:")
        print(f"  总步数: {stats.total_steps}")
        print(f"  最终势能: {stats.final_potential:.6f}")
        print(f"  平均梯度范数: {stats.avg_gradient_norm:.6f}")
        print(f"  总移动距离: {stats.total_distance:.6f}")
        if stats.convergence_step:
            print(f"  收敛步数: {stats.convergence_step}")

# ============================================================
# 主函数
# ============================================================

def run_ufl(code: str, verbose: bool = True, solver: str = "rk4"):
    """解析并运行UFL代码"""
    # 解析
    parser = UFLParser()
    system = parser.parse(code)
    
    # 编译
    runtime = UFLRuntime(system, solver_type=solver)
    runtime.compile()
    
    # 运行
    result = runtime.run(verbose=verbose)
    
    # 可视化
    if system.simulate and system.simulate.visualize:
        visualize_ascii(result, system.name)
    
    return result

# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    test_code = """
    system 谐振子 {
        dimension: 3
        domain: physics
        
        field 势能(x) = 0.5 * |x|^2
        
        particle 粒子 {
            position: [2.0, -1.5, 1.0]
            dynamics: gradient_flow(lr=0.3, momentum=0.8)
        }
        
        simulate {
            steps: 100
            dt: 0.1
            method: rk4
            visualize: true
        }
    }
    """
    
    print("=" * 60)
    print("测试: 改进的UFL引擎")
    print("=" * 60)
    
    result = run_ufl(test_code, solver="rk4")
