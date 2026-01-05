#!/usr/bin/env python3
"""
UFL 快速开始脚本
"""

from engine import run_ufl

print("\n" + "="*60)
print("UFL 快速开始演示")
print("="*60)

# 示例1：谐振子
print("\n[示例1] 谐振子")
code1 = """
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
        visualize: false
    }
}
"""

result1 = run_ufl(code1, verbose=False)
print(f"✓ 粒子从 {result1['particles'][0].trajectory[0]} 收敛到 {result1['particles'][0].trajectory[-1]}")

# 示例2：参数优化
print("\n[示例2] 参数优化")
code2 = """
system 参数优化 {
    dimension: 3
    domain: optimization
    
    field 损失(x) = |x|^2
    
    particle 权重 {
        position: [1.0, -1.0, 0.5]
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

result2 = run_ufl(code2, verbose=False)
final = result2['particles'][0].trajectory[-1]
print(f"✓ 参数收敛到最优值，最终位置: {final}")

# 示例3：多粒子
print("\n[示例3] 多粒子系统")
code3 = """
system 多粒子 {
    dimension: 2
    domain: physics
    
    field 势能(x) = 0.5 * |x|^2
    
    particle 粒子1 {
        position: [2.0, 0.0]
        dynamics: gradient_flow(lr=0.1, momentum=0.8)
    }
    
    particle 粒子2 {
        position: [0.0, 2.0]
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

result3 = run_ufl(code3, verbose=False)
print(f"✓ 粒子1最终位置: {result3['particles'][0].trajectory[-1]}")
print(f"✓ 粒子2最终位置: {result3['particles'][1].trajectory[-1]}")

print("\n" + "="*60)
print("✓ 快速开始演示完成！")
print("="*60)
print("\n更多示例请查看 examples.py")
print("详细文档请查看 GUIDE.md")
