"""
UFL 单元测试套件

测试覆盖：
- 语法解析
- 表达式编译
- 数值求解
- 约束系统
- 完整的模拟
"""

import unittest
import numpy as np
from syntax import (
    UFLParser, UFLSyntaxError, UFLTypeError, SystemDef,
    Expr, BinaryOp, UnaryOp, Variable, Number
)
from engine import (
    ExpressionCompiler, SymbolicDifferentiator, UFLRuntime,
    EulerSolver, RK4Solver, SphereConstraint, run_ufl
)

# ============================================================
# 语法解析测试
# ============================================================

class TestSyntaxParsing(unittest.TestCase):
    """测试语法解析"""
    
    def setUp(self):
        self.parser = UFLParser()
    
    def test_simple_system(self):
        """测试简单系统解析"""
        code = """
        system 测试 {
            dimension: 2
            domain: physics
        }
        """
        ast = self.parser.parse(code)
        self.assertEqual(ast.name, "测试")
        self.assertEqual(ast.dimension, 2)
        self.assertEqual(ast.domain, "physics")
    
    def test_field_parsing(self):
        """测试场定义解析"""
        code = """
        system 测试 {
            field 势能(x) = 0.5 * |x|^2
        }
        """
        ast = self.parser.parse(code)
        self.assertEqual(len(ast.fields), 1)
        self.assertEqual(ast.fields[0].name, "势能")
        self.assertEqual(ast.fields[0].variable, "x")
    
    def test_particle_parsing(self):
        """测试粒子定义解析"""
        code = """
        system 测试 {
            particle 粒子1 {
                position: [1.0, 2.0, 3.0]
                velocity: [0.1, 0.2, 0.3]
                dynamics: gradient_flow(lr=0.1, momentum=0.9)
            }
        }
        """
        ast = self.parser.parse(code)
        self.assertEqual(len(ast.particles), 1)
        p = ast.particles[0]
        self.assertEqual(p.name, "粒子1")
        np.testing.assert_array_almost_equal(p.position, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(p.velocity, [0.1, 0.2, 0.3])
        self.assertEqual(p.dynamics, "gradient_flow")
        self.assertAlmostEqual(p.dynamics_params['lr'], 0.1)
        self.assertAlmostEqual(p.dynamics_params['momentum'], 0.9)
    
    def test_simulate_parsing(self):
        """测试模拟配置解析"""
        code = """
        system 测试 {
            simulate {
                steps: 100
                dt: 0.05
                method: rk4
                visualize: true
            }
        }
        """
        ast = self.parser.parse(code)
        self.assertIsNotNone(ast.simulate)
        self.assertEqual(ast.simulate.steps, 100)
        self.assertAlmostEqual(ast.simulate.dt, 0.05)
        self.assertEqual(ast.simulate.method, "rk4")
        self.assertTrue(ast.simulate.visualize)
    
    def test_syntax_error_handling(self):
        """测试语法错误处理"""
        code = """
        system 测试 {
            dimension 2
        }
        """
        with self.assertRaises(UFLSyntaxError):
            self.parser.parse(code)
    
    def test_expression_parsing(self):
        """测试表达式解析"""
        code = """
        system 测试 {
            field V(x) = 0.5 * |x|^2 + x[0]
        }
        """
        ast = self.parser.parse(code)
        self.assertEqual(len(ast.fields), 1)
        # 检查表达式被正确解析
        self.assertIsNotNone(ast.fields[0].expression)

# ============================================================
# 表达式编译测试
# ============================================================

class TestExpressionCompilation(unittest.TestCase):
    """测试表达式编译"""
    
    def test_compile_simple_expression(self):
        """测试简单表达式编译"""
        # 创建表达式 0.5 * |x|^2
        expr = BinaryOp(
            op='*',
            left=Number(value=0.5),
            right=BinaryOp(
                op='^',
                left=UnaryOp(op='|·|', operand=Variable(name='x')),
                right=Number(value=2.0)
            )
        )
        
        func = ExpressionCompiler.compile_expr(expr, 3)
        
        # 测试
        x = np.array([1.0, 0.0, 0.0])
        result = func(x)
        self.assertAlmostEqual(result, 0.5)
        
        x = np.array([1.0, 1.0, 1.0])
        result = func(x)
        self.assertAlmostEqual(result, 1.5)
    
    def test_compile_addition(self):
        """测试加法表达式"""
        expr = BinaryOp(
            op='+',
            left=Number(value=1.0),
            right=Number(value=2.0)
        )
        
        func = ExpressionCompiler.compile_expr(expr, 1)
        result = func(np.array([0.0]))
        self.assertAlmostEqual(result, 3.0)

# ============================================================
# 符号微分测试
# ============================================================

class TestSymbolicDifferentiation(unittest.TestCase):
    """测试符号微分"""
    
    def test_diff_constant(self):
        """测试常数微分"""
        expr = Number(value=5.0)
        deriv = SymbolicDifferentiator.diff(expr, 'x')
        self.assertIsInstance(deriv, Number)
        self.assertEqual(deriv.value, 0.0)
    
    def test_diff_variable(self):
        """测试变量微分"""
        expr = Variable(name='x')
        deriv = SymbolicDifferentiator.diff(expr, 'x')
        self.assertIsInstance(deriv, Number)
        self.assertEqual(deriv.value, 1.0)
    
    def test_diff_sum(self):
        """测试和的微分"""
        expr = BinaryOp(
            op='+',
            left=Variable(name='x'),
            right=Number(value=2.0)
        )
        deriv = SymbolicDifferentiator.diff(expr, 'x')
        # 应该得到 1 + 0 = 1
        self.assertIsInstance(deriv, BinaryOp)
    
    def test_diff_product(self):
        """测试乘积的微分"""
        # d/dx (x * x) = 2x
        expr = BinaryOp(
            op='*',
            left=Variable(name='x'),
            right=Variable(name='x')
        )
        deriv = SymbolicDifferentiator.diff(expr, 'x')
        self.assertIsInstance(deriv, BinaryOp)

# ============================================================
# 数值求解器测试
# ============================================================

class TestNumericalSolvers(unittest.TestCase):
    """测试数值求解器"""
    
    def test_euler_solver(self):
        """测试欧拉求解器"""
        solver = EulerSolver()
        
        # 简单势能：V(x) = x^2
        def potential(x):
            return np.sum(x**2)
        
        x = np.array([1.0, 0.0])
        v = np.array([0.0, 0.0])
        
        x_new, v_new = solver.step(x, v, potential, dt=0.1, lr=0.1, momentum=0.8)
        
        # 应该向原点移动
        self.assertLess(np.linalg.norm(x_new), np.linalg.norm(x))
    
    def test_rk4_solver(self):
        """测试RK4求解器"""
        solver = RK4Solver()
        
        def potential(x):
            return np.sum(x**2)
        
        x = np.array([1.0, 0.0])
        v = np.array([0.0, 0.0])
        
        x_new, v_new = solver.step(x, v, potential, dt=0.1, lr=0.1, momentum=0.8)
        
        # 应该向原点移动
        self.assertLess(np.linalg.norm(x_new), np.linalg.norm(x))

# ============================================================
# 约束系统测试
# ============================================================

class TestConstraints(unittest.TestCase):
    """测试约束系统"""
    
    def test_sphere_constraint(self):
        """测试球面约束"""
        constraint = SphereConstraint(radius=1.0)
        
        x = np.array([2.0, 0.0, 0.0])
        x_constrained = constraint.apply(x)
        
        # 应该在单位球面上
        norm = np.linalg.norm(x_constrained)
        self.assertAlmostEqual(norm, 1.0)
    
    def test_sphere_constraint_gradient_projection(self):
        """测试球面约束梯度投影"""
        constraint = SphereConstraint(radius=1.0)
        
        x = np.array([1.0, 0.0, 0.0])
        grad = np.array([1.0, 1.0, 1.0])
        
        grad_proj = constraint.project_gradient(grad, x)
        
        # 投影后的梯度应该垂直于x
        dot_product = np.dot(grad_proj, x)
        self.assertAlmostEqual(dot_product, 0.0, places=5)

# ============================================================
# 完整模拟测试
# ============================================================

class TestCompleteSimulation(unittest.TestCase):
    """测试完整的模拟"""
    
    def test_harmonic_oscillator(self):
        """测试谐振子"""
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
                steps: 50
                dt: 0.1
                method: rk4
                visualize: false
            }
        }
        """
        
        result = run_ufl(code, verbose=False)
        
        # 检查粒子是否移动
        particles = result["particles"]
        self.assertEqual(len(particles), 1)
        
        initial_pos = particles[0].trajectory[0]
        final_pos = particles[0].trajectory[-1]
        
        # 应该接近原点
        self.assertLess(np.linalg.norm(final_pos), np.linalg.norm(initial_pos))
    
    def test_double_well_potential(self):
        """测试双势阱"""
        code = """
        system 双势阱 {
            dimension: 1
            domain: physics
            
            field 势能(x) = |x|^4 - 2 * |x|^2
            
            particle 粒子 {
                position: [0.5]
                dynamics: gradient_flow(lr=0.1, momentum=0.8)
            }
            
            simulate {
                steps: 100
                dt: 0.1
                method: euler
                visualize: false
            }
        }
        """
        
        result = run_ufl(code, verbose=False)
        particles = result["particles"]
        
        # 应该收敛到某个最小值
        final_pos = particles[0].trajectory[-1]
        self.assertIsNotNone(final_pos)
    
    def test_optimization_domain(self):
        """测试优化领域"""
        code = """
        system 参数优化 {
            dimension: 3
            domain: optimization
            
            field 损失(x) = |x|^2
            
            particle 参数 {
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
        
        result = run_ufl(code, verbose=False)
        particles = result["particles"]
        
        # 参数应该接近零
        final_pos = particles[0].trajectory[-1]
        self.assertLess(np.linalg.norm(final_pos), 0.1)

# ============================================================
# 性能测试
# ============================================================

class TestPerformance(unittest.TestCase):
    """测试性能"""
    
    def test_high_dimensional_system(self):
        """测试高维系统"""
        code = """
        system 高维 {
            dimension: 10
            domain: optimization
            
            field 损失(x) = |x|^2
            
            particle 参数 {
                position: [1.0, -1.0, 0.5, 0.2, -0.3, 0.1, -0.2, 0.3, -0.1, 0.2]
                dynamics: gradient_flow(lr=0.1, momentum=0.8)
            }
            
            simulate {
                steps: 20
                dt: 0.1
                method: euler
                visualize: false
            }
        }
        """
        
        result = run_ufl(code, verbose=False)
        self.assertIsNotNone(result)
    
    def test_multiple_particles(self):
        """测试多粒子系统"""
        code = """
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
                steps: 30
                dt: 0.1
                method: euler
                visualize: false
            }
        }
        """
        
        result = run_ufl(code, verbose=False)
        particles = result["particles"]
        self.assertEqual(len(particles), 2)

# ============================================================
# 运行测试
# ============================================================

if __name__ == '__main__':
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试
    suite.addTests(loader.loadTestsFromTestCase(TestSyntaxParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestExpressionCompilation))
    suite.addTests(loader.loadTestsFromTestCase(TestSymbolicDifferentiation))
    suite.addTests(loader.loadTestsFromTestCase(TestNumericalSolvers))
    suite.addTests(loader.loadTestsFromTestCase(TestConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestCompleteSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✓ 所有测试通过！")
    else:
        print("\n✗ 有测试失败")
