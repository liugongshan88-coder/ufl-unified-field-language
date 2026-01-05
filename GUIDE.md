# UFL（统一场语言）使用指南

**版本**: 2.0（改进版）  
**作者**: Manus AI  
**日期**: 2026年1月

---

## 目录

1. [快速开始](#快速开始)
2. [语言语法](#语言语法)
3. [表达式](#表达式)
4. [场景示例](#场景示例)
5. [高级特性](#高级特性)
6. [API参考](#api参考)
7. [常见问题](#常见问题)

---

## 快速开始

### 最简单的例子

```python
from engine import run_ufl

code = """
system 谐振子 {
    dimension: 2
    domain: physics
    
    field 势能(x) = 0.5 * |x|^2
    
    particle 粒子 {
        position: [1.0, 0.0]
        dynamics: gradient_flow(lr=0.1, momentum=0.8)
    }
    
    simulate {
        steps: 100
        dt: 0.1
        method: rk4
        visualize: true
    }
}
"""

result = run_ufl(code)
```

**说明**：
- `dimension: 2` - 2维空间
- `domain: physics` - 物理领域
- `field 势能(x) = 0.5 * |x|^2` - 定义势能函数
- `particle 粒子` - 定义一个粒子
- `position: [1.0, 0.0]` - 初始位置
- `dynamics: gradient_flow(...)` - 使用梯度流动力学
- `simulate {...}` - 模拟配置

---

## 语言语法

### 系统定义

```ufl
system <名称> {
    dimension: <整数>
    domain: <physics|optimization|cognitive>
    
    // 场定义
    field <名称>(x) = <表达式>
    
    // 粒子定义
    particle <名称> {
        position: [x, y, z, ...]
        velocity: [vx, vy, vz, ...]  // 可选
        dynamics: <动力学类型>(<参数>)
    }
    
    // 约束定义（可选）
    constraint <名称> {
        type: <约束类型>
        <参数>: <值>
    }
    
    // 模拟配置
    simulate {
        steps: <整数>
        dt: <浮点数>
        method: <euler|rk4|adaptive>
        visualize: <true|false>
    }
}
```

### 域（Domain）

| 域 | 用途 | 示例 |
|---|------|------|
| **physics** | 物理系统模拟 | 粒子、波、流体 |
| **optimization** | 参数优化 | 神经网络训练、参数调优 |
| **cognitive** | 认知系统 | 信念更新、概念漂移 |

### 动力学（Dynamics）

```ufl
// 梯度流
dynamics: gradient_flow(lr=0.1, momentum=0.8)

// 参数说明：
// - lr: 学习率（步长）
// - momentum: 动量系数（0-1）
```

### 约束（Constraint）

```ufl
// 球面约束
constraint 球面 {
    type: sphere
    radius: 1.0
}

// 圆柱约束
constraint 圆柱 {
    type: cylinder
    radius: 1.0
}

// 平面约束
constraint 平面 {
    type: plane
    normal: [0, 0, 1]
    distance: 0.0
}
```

---

## 表达式

### 支持的操作

| 操作 | 符号 | 例子 |
|------|------|------|
| 加法 | `+` | `x + y` |
| 减法 | `-` | `x - y` |
| 乘法 | `*` | `2 * x` |
| 除法 | `/` | `x / 2` |
| 幂 | `^` | `x^2` |
| 范数 | `\|x\|` | `\|x\|` |
| 范数平方 | `\|x\|^2` | `\|x\|^2` |

### 函数

```ufl
sqrt(x)     // 平方根
abs(x)      // 绝对值
sin(x)      // 正弦
cos(x)      // 余弦
exp(x)      // 指数
log(x)      // 自然对数
```

### 向量操作

```ufl
// 向量范数
|x|         // L2范数
|x|^2       // 范数平方

// 向量分量
x[0]        // 第一个分量
x[1]        // 第二个分量
```

### 优先级

从高到低：
1. 括号 `()`
2. 一元操作 `-`, `|·|`
3. 幂 `^`
4. 乘法/除法 `*`, `/`
5. 加法/减法 `+`, `-`

---

## 场景示例

### 例1：谐振子（物理）

```ufl
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
```

**说明**：粒子从 [2, -1.5, 1] 开始，在二次势能 V(x) = 0.5|x|² 的作用下，沿梯度流向原点。

### 例2：双势阱（物理）

```ufl
system 双势阱 {
    dimension: 2
    domain: physics
    
    field 势能(x) = |x|^4 - 2 * |x|^2
    
    particle 粒子 {
        position: [1.5, 0.5]
        dynamics: gradient_flow(lr=0.1, momentum=0.9)
    }
    
    simulate {
        steps: 80
        dt: 0.1
        method: rk4
        visualize: true
    }
}
```

**说明**：粒子在双势阱中运动。势能有两个最小值，粒子会收敛到其中之一。

### 例3：参数优化（优化）

```ufl
system 神经网络训练 {
    dimension: 5
    domain: optimization
    
    field 损失(x) = |x|^2
    
    particle 权重 {
        position: [1.0, -2.0, 3.0, -1.5, 2.5]
        dynamics: gradient_flow(lr=0.2, momentum=0.85)
    }
    
    simulate {
        steps: 60
        dt: 0.1
        method: rk4
        visualize: true
    }
}
```

**说明**：参数向量从初始值开始，沿损失函数的梯度下降，收敛到最优解（原点）。

### 例4：概念漂移（认知）

```ufl
system 概念漂移 {
    dimension: 3
    domain: cognitive
    
    field 注意力(x) = 2.0 * |x|^2
    
    particle 猫的概念 {
        position: [1.0, 0.2, 0.8]
        dynamics: gradient_flow(lr=0.15, momentum=0.8)
    }
    
    simulate {
        steps: 50
        dt: 0.1
        method: rk4
        visualize: true
    }
}
```

**说明**：概念在语义空间中漂移，被注意力吸引。

### 例5：机器人导航（应用）

```python
from robot_sim_3d import UFL_World_3D

# 创建3D世界
world = UFL_World_3D(dimension=3)
world.setup_scenario("maze_3d", dimension=3)

# 运行模拟
success, result = world.run(max_steps=500)

# 查看结果
print(f"成功: {success}")
print(f"步数: {result['steps']}")
print(f"轨迹长度: {result['trajectory_length']:.3f}")
print(f"平滑度: {result['smoothness']:.3f}")
```

---

## 高级特性

### 多粒子系统

```ufl
system 多粒子 {
    dimension: 2
    domain: physics
    
    field 势能(x) = 0.5 * |x|^2
    
    particle 粒子1 {
        position: [1.0, 0.0]
        dynamics: gradient_flow(lr=0.1, momentum=0.8)
    }
    
    particle 粒子2 {
        position: [0.0, 1.0]
        dynamics: gradient_flow(lr=0.1, momentum=0.8)
    }
    
    simulate {
        steps: 100
        dt: 0.1
        method: rk4
        visualize: true
    }
}
```

### 多势场

```ufl
system 多势场 {
    dimension: 2
    domain: physics
    
    field 吸引(x) = 0.5 * |x|^2
    field 排斥(x) = 1.0 / |x|
    
    particle 粒子 {
        position: [2.0, 0.0]
        dynamics: gradient_flow(lr=0.1, momentum=0.8)
    }
    
    simulate {
        steps: 100
        dt: 0.1
        visualize: true
    }
}
```

**说明**：多个势场会自动叠加。总势能为 V_total = V_吸引 + V_排斥。

### 约束系统

```ufl
system 约束系统 {
    dimension: 3
    domain: physics
    
    field 势能(x) = 0.5 * |x|^2
    
    particle 粒子 {
        position: [2.0, 0.0, 0.0]
        dynamics: gradient_flow(lr=0.1, momentum=0.8)
    }
    
    constraint 球面 {
        type: sphere
        radius: 1.0
    }
    
    simulate {
        steps: 100
        dt: 0.1
        visualize: true
    }
}
```

**说明**：粒子被限制在单位球面上运动。

### 不同求解器

```ufl
// 欧拉方法（快速但不精确）
simulate {
    steps: 100
    dt: 0.1
    method: euler
}

// RK4方法（精确但较慢）
simulate {
    steps: 100
    dt: 0.1
    method: rk4
}

// 自适应步长（自动调整步长）
simulate {
    steps: 100
    dt: 0.1
    method: adaptive
}
```

---

## API参考

### 运行UFL代码

```python
from engine import run_ufl

result = run_ufl(
    code: str,           # UFL代码
    verbose: bool = True,  # 是否打印详细信息
    solver: str = "rk4"    # 求解器类型
)
```

**返回值**：
```python
{
    "particles": [ParticleState, ...],
    "history": {
        "potential": [float, ...],
        "step": [int, ...],
        "gradient_norm": [float, ...]
    },
    "stats": SimulationStats
}
```

### 3D机器人模拟

```python
from robot_sim_3d import UFL_World_3D

world = UFL_World_3D(dimension=3)
world.setup_scenario("simple", dimension=3)
success, result = world.run(max_steps=500)
```

### 性能指标

```python
from robot_sim_3d import calculate_metrics

metrics = calculate_metrics(world, result)
print(metrics)
# 输出：
# - success: 是否成功到达目标
# - steps: 所需步数
# - trajectory_length: 轨迹总长度
# - smoothness: 轨迹平滑度
# - final_distance: 最终距离
# - efficiency: 效率（直线距离/实际距离）
```

---

## 常见问题

### Q1：如何调整收敛速度？

**A**：调整学习率 `lr` 和动量 `momentum`：
- 增大 `lr` → 收敛更快但可能不稳定
- 增大 `momentum` → 更平滑但可能振荡

```ufl
dynamics: gradient_flow(lr=0.5, momentum=0.9)  // 快速收敛
dynamics: gradient_flow(lr=0.05, momentum=0.5) // 稳定收敛
```

### Q2：如何处理局部最小值？

**A**：
1. 添加噪声（随机初始化）
2. 使用多个粒子从不同位置开始
3. 调整势函数

### Q3：如何可视化高维系统？

**A**：使用降维投影（PCA、t-SNE等）或查看轨迹统计：

```python
result = run_ufl(code)
potentials = result["history"]["potential"]
distances = result["history"]["gradient_norm"]
# 绘制这些曲线
```

### Q4：为什么粒子不动？

**A**：可能的原因：
1. 初始位置已在最小值
2. 学习率太小
3. 势函数在该点的梯度为零

### Q5：如何在约束流形上运动？

**A**：使用约束定义：

```ufl
constraint 球面 {
    type: sphere
    radius: 1.0
}
```

粒子会自动投影到球面上。

### Q6：支持多少维？

**A**：理论上无限制，但：
- 2-3维：可视化友好
- 4-10维：优化常见
- 100+维：需要特殊处理

---

## 最佳实践

1. **从简单开始**：先测试2D系统，再升级到3D
2. **调整参数**：根据收敛情况调整 `lr` 和 `momentum`
3. **监控势能**：检查势能是否单调递减
4. **验证约束**：确保约束被正确应用
5. **使用适当的求解器**：
   - 快速测试：`euler`
   - 精确计算：`rk4`
   - 自动调整：`adaptive`

---

## 更多资源

- 数学基础：见 `UFL_Mathematical_Foundations.md`
- 系统分析：见 `UFL_System_Analysis.md`
- 测试示例：见 `test_ufl.py`
- 3D模拟：见 `robot_sim_3d.py`

---

*本指南涵盖了UFL的主要功能。如有问题，请参考源代码中的详细注释。*
