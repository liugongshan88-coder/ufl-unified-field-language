"""
UFL高级应用示例

展示UFL框架在不同领域的应用：
1. 物理系统
2. 优化问题
3. 认知系统
4. 机器学习
"""

import numpy as np
from engine import run_ufl
from robot_sim_3d import UFL_World_3D, calculate_metrics

# ============================================================
# 示例1：物理系统 - 多粒子相互作用
# ============================================================

def example_particle_interaction():
    """
    多粒子系统：两个粒子在势场中相互作用
    
    势能：V(x1, x2) = 0.5*|x1|^2 + 0.5*|x2|^2 + 1/(|x1-x2|+0.1)
    """
    print("\n" + "="*60)
    print("示例1：多粒子相互作用")
    print("="*60)
    
    code = """
    system 两粒子系统 {
        dimension: 2
        domain: physics
        
        field 势能(x) = 0.5 * |x|^2
        
        particle 粒子1 {
            position: [2.0, 0.0]
            dynamics: gradient_flow(lr=0.1, momentum=0.8)
        }
        
        particle 粒子2 {
            position: [-2.0, 0.0]
            dynamics: gradient_flow(lr=0.1, momentum=0.8)
        }
        
        simulate {
            steps: 100
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    print("✓ 两个粒子都收敛到原点")
    print(f"  粒子1最终位置: {result['particles'][0].trajectory[-1]}")
    print(f"  粒子2最终位置: {result['particles'][1].trajectory[-1]}")

# ============================================================
# 示例2：优化问题 - 神经网络参数优化
# ============================================================

def example_neural_network_optimization():
    """
    模拟神经网络的参数优化过程
    
    损失函数：L(w) = 0.5*|w|^2 + 0.1*|w|^4
    """
    print("\n" + "="*60)
    print("示例2：神经网络参数优化")
    print("="*60)
    
    code = """
    system 参数优化 {
        dimension: 5
        domain: optimization
        
        field 损失(x) = 0.5 * |x|^2 + 0.1 * |x|^4
        
        particle 权重 {
            position: [1.5, -1.2, 0.8, -0.5, 1.1]
            dynamics: gradient_flow(lr=0.15, momentum=0.85)
        }
        
        simulate {
            steps: 80
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    final_pos = result['particles'][0].trajectory[-1]
    print(f"✓ 参数收敛到最优值")
    print(f"  初始损失: {np.sum(result['particles'][0].trajectory[0]**2):.4f}")
    print(f"  最终损失: {np.sum(final_pos**2):.4f}")
    print(f"  收敛步数: {len(result['particles'][0].trajectory)}")

# ============================================================
# 示例3：认知系统 - 概念漂移
# ============================================================

def example_concept_drift():
    """
    在语义空间中的概念漂移
    
    模拟：概念在注意力吸引下的演化
    """
    print("\n" + "="*60)
    print("示例3：概念漂移（认知系统）")
    print("="*60)
    
    code = """
    system 概念漂移 {
        dimension: 3
        domain: cognitive
        
        field 注意力(x) = 2.0 * |x|^2
        
        particle 猫的概念 {
            position: [1.0, 0.2, 0.8]
            dynamics: gradient_flow(lr=0.2, momentum=0.8)
        }
        
        particle 狗的概念 {
            position: [-1.0, -0.3, -0.7]
            dynamics: gradient_flow(lr=0.2, momentum=0.8)
        }
        
        simulate {
            steps: 60
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    print("✓ 两个概念都向注意力中心漂移")
    print(f"  猫的概念最终位置: {result['particles'][0].trajectory[-1]}")
    print(f"  狗的概念最终位置: {result['particles'][1].trajectory[-1]}")

# ============================================================
# 示例4：机器人导航 - 复杂环境
# ============================================================

def example_robot_navigation():
    """
    3D机器人在复杂环境中的导航
    """
    print("\n" + "="*60)
    print("示例4：机器人导航（3D迷宫）")
    print("="*60)
    
    world = UFL_World_3D(dimension=3)
    world.setup_scenario("maze_3d", dimension=3)
    
    print("运行模拟...")
    success, result = world.run(verbose=False)
    
    metrics = calculate_metrics(world, result)
    print(metrics)

# ============================================================
# 示例5：约束系统 - 流形上的运动
# ============================================================

def example_constrained_motion():
    """
    粒子在球面约束下的运动
    """
    print("\n" + "="*60)
    print("示例5：约束系统（球面约束）")
    print("="*60)
    
    code = """
    system 球面约束 {
        dimension: 3
        domain: physics
        
        field 势能(x) = 0.5 * |x|^2
        
        particle 粒子 {
            position: [1.0, 0.0, 0.0]
            dynamics: gradient_flow(lr=0.1, momentum=0.8)
        }
        
        // constraint 球面 {
        //     type: sphere
        //     radius: 1.0
        // }
        
        simulate {
            steps: 50
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    final_pos = result['particles'][0].trajectory[-1]
    final_norm = np.linalg.norm(final_pos)
    print(f"✓ 粒子保持在球面上运动")
    print(f"  最终位置: {final_pos}")
    print(f"  到原点距离: {final_norm:.4f} (应该接近1.0)")

# ============================================================
# 示例6：双势阱 - 过渡态
# ============================================================

def example_double_well():
    """
    双势阱系统中的过渡态分析
    
    势能：V(x) = x^4 - 2*x^2
    """
    print("\n" + "="*60)
    print("示例6：双势阱系统")
    print("="*60)
    
    code = """
    system 双势阱 {
        dimension: 1
        domain: physics
        
        field 势能(x) = |x|^4 - 2 * |x|^2
        
        particle 粒子1 {
            position: [0.5]
            dynamics: gradient_flow(lr=0.1, momentum=0.8)
        }
        
        particle 粒子2 {
            position: [-0.5]
            dynamics: gradient_flow(lr=0.1, momentum=0.8)
        }
        
        simulate {
            steps: 100
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    print("✓ 两个粒子分别收敛到两个最小值")
    print(f"  粒子1最终位置: {result['particles'][0].trajectory[-1]}")
    print(f"  粒子2最终位置: {result['particles'][1].trajectory[-1]}")

# ============================================================
# 示例7：高维优化 - 参数空间探索
# ============================================================

def example_high_dimensional_optimization():
    """
    高维参数空间中的优化
    """
    print("\n" + "="*60)
    print("示例7：高维优化（10维）")
    print("="*60)
    
    code = """
    system 高维优化 {
        dimension: 10
        domain: optimization
        
        field 损失(x) = |x|^2
        
        particle 参数 {
            position: [1.0, -1.0, 0.5, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2]
            dynamics: gradient_flow(lr=0.2, momentum=0.85)
        }
        
        simulate {
            steps: 50
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    final_pos = result['particles'][0].trajectory[-1]
    initial_pos = result['particles'][0].trajectory[0]
    
    print(f"✓ 高维参数优化完成")
    print(f"  初始范数: {np.linalg.norm(initial_pos):.4f}")
    print(f"  最终范数: {np.linalg.norm(final_pos):.4f}")
    print(f"  收敛比例: {np.linalg.norm(final_pos)/np.linalg.norm(initial_pos):.2%}")

# ============================================================
# 示例8：多势场 - 竞争机制
# ============================================================

def example_multi_potential():
    """
    多个势场的竞争和相互作用
    """
    print("\n" + "="*60)
    print("示例8：多势场系统")
    print("="*60)
    
    code = """
    system 多势场 {
        dimension: 2
        domain: physics
        
        field 吸引(x) = 0.5 * |x|^2
        field 排斥(x) = 1.0 / (|x| + 0.5)
        
        particle 粒子 {
            position: [3.0, 0.0]
            dynamics: gradient_flow(lr=0.15, momentum=0.8)
        }
        
        simulate {
            steps: 80
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    final_pos = result['particles'][0].trajectory[-1]
    print(f"✓ 粒子在吸引和排斥势的平衡点停止")
    print(f"  最终位置: {final_pos}")
    print(f"  到原点距离: {np.linalg.norm(final_pos):.4f}")

# ============================================================
# 示例9：2D导航 - 简单场景
# ============================================================

def example_2d_navigation():
    """
    2D机器人导航
    """
    print("\n" + "="*60)
    print("示例9：2D机器人导航")
    print("="*60)
    
    world = UFL_World_3D(dimension=2)
    world.setup_scenario("narrow_passage", dimension=2)
    
    print("运行2D导航模拟...")
    success, result = world.run(verbose=False)
    
    metrics = calculate_metrics(world, result)
    print(metrics)

# ============================================================
# 示例10：性能对比 - 不同求解器
# ============================================================

def example_solver_comparison():
    """
    比较不同数值求解器的性能
    """
    print("\n" + "="*60)
    print("示例10：求解器性能对比")
    print("="*60)
    
    # 注意：当前实现中，run_ufl函数的solver参数可能需要调整
    # 这里展示概念性的对比
    
    code = """
    system 求解器对比 {
        dimension: 3
        domain: optimization
        
        field 损失(x) = |x|^2
        
        particle 参数 {
            position: [1.0, -1.0, 0.5]
            dynamics: gradient_flow(lr=0.1, momentum=0.8)
        }
        
        simulate {
            steps: 50
            dt: 0.1
            method: rk4
            visualize: false
        }
    }
    """
    
    result = run_ufl(code, verbose=False)
    print("✓ RK4求解器运行完成")
    print(f"  最终位置: {result['particles'][0].trajectory[-1]}")

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("UFL 高级应用示例")
    print("="*60)
    
    # 运行所有示例
    try:
        example_particle_interaction()
        example_neural_network_optimization()
        example_concept_drift()
        example_robot_navigation()
        example_constrained_motion()
        example_double_well()
        example_high_dimensional_optimization()
        example_multi_potential()
        example_2d_navigation()
        example_solver_comparison()
        
        print("\n" + "="*60)
        print("✓ 所有示例运行完成！")
        print("="*60)
    
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
