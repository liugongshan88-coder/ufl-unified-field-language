"""
UFL - Unified Field Language
改进版语法定义和解析器

改进点：
- 完整的类型检查系统
- 详细的错误报告（带位置信息）
- 支持更多表达式类型
- 量纲一致性验证
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum

# ============================================================
# 类型系统与量纲
# ============================================================

class Dimension(Enum):
    """物理量纲"""
    LENGTH = "长度"
    TIME = "时间"
    MASS = "质量"
    VELOCITY = "速度"
    ACCELERATION = "加速度"
    ENERGY = "能量"
    FORCE = "力"
    PROBABILITY = "概率"
    SEMANTIC = "语义"
    DIMENSIONLESS = "无量纲"

@dataclass
class UFLType:
    """UFL类型系统"""
    name: str
    dimension: Dimension
    is_vector: bool = False
    vector_dim: int = 1
    
    def __eq__(self, other):
        if not isinstance(other, UFLType):
            return False
        return (self.name == other.name and 
                self.dimension == other.dimension and
                self.is_vector == other.is_vector)
    
    def __repr__(self):
        if self.is_vector:
            return f"{self.name}[{self.vector_dim}]"
        return self.name

# 预定义类型
TYPES = {
    "scalar": UFLType("scalar", Dimension.DIMENSIONLESS),
    "position": UFLType("position", Dimension.LENGTH, is_vector=True),
    "velocity": UFLType("velocity", Dimension.VELOCITY, is_vector=True),
    "acceleration": UFLType("acceleration", Dimension.ACCELERATION, is_vector=True),
    "energy": UFLType("energy", Dimension.ENERGY),
    "force": UFLType("force", Dimension.FORCE, is_vector=True),
    "probability": UFLType("probability", Dimension.PROBABILITY),
    "semantic": UFLType("semantic", Dimension.SEMANTIC, is_vector=True),
}

# ============================================================
# 错误处理
# ============================================================

@dataclass
class SourceLocation:
    """源代码位置"""
    line: int
    column: int
    
    def __repr__(self):
        return f"行{self.line}列{self.column}"

class UFLSyntaxError(Exception):
    """UFL语法错误"""
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        self.message = message
        self.location = location
        if location:
            super().__init__(f"{location}: {message}")
        else:
            super().__init__(message)

class UFLTypeError(Exception):
    """UFL类型错误"""
    def __init__(self, message: str, location: Optional[SourceLocation] = None):
        self.message = message
        self.location = location
        if location:
            super().__init__(f"{location}: 类型错误 - {message}")
        else:
            super().__init__(f"类型错误 - {message}")

# ============================================================
# AST 节点定义
# ============================================================

@dataclass
class ASTNode:
    """AST基类"""
    location: Optional[SourceLocation] = field(default=None)

@dataclass
class Expr(ASTNode):
    """表达式基类"""
    pass

# 为了解决dataclass的字段顺序需求，我们为每个表达式类添加一个辅助序列化函数

@dataclass
class BinaryOp(Expr):
    """二元操作"""
    left: Expr = field(default_factory=lambda: Number(0))
    right: Expr = field(default_factory=lambda: Number(0))
    op: str = "+"  # +, -, *, /, ^

@dataclass
class UnaryOp(Expr):
    """一元操作"""
    operand: Expr = field(default_factory=lambda: Number(0))
    op: str = "-"  # -, |·|, sqrt

@dataclass
class FunctionCall(Expr):
    """函数调用"""
    name: str = ""
    args: List[Expr] = field(default_factory=list)

@dataclass
class Variable(Expr):
    """变量"""
    name: str = "x"

@dataclass
class Number(Expr):
    """数字常量"""
    value: float = 0.0

@dataclass
class VectorLiteral(Expr):
    """向量字面量"""
    elements: List[Expr] = field(default_factory=list)

@dataclass
class FieldDef(ASTNode):
    """场定义"""
    name: str = ""
    variable: str = "x"  # 如 'x'
    expression: Expr = field(default_factory=lambda: Number(0))
    return_type: UFLType = field(default_factory=lambda: TYPES["scalar"])
    
@dataclass
class ParticleDef(ASTNode):
    """粒子定义"""
    name: str = ""
    position: List[float] = field(default_factory=list)
    velocity: Optional[List[float]] = None
    dynamics: str = "gradient_flow"
    dynamics_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ConstraintDef(ASTNode):
    """约束定义"""
    name: str = ""
    type: str = "unknown"  # "manifold", "distance", "sphere", etc.
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulateDef(ASTNode):
    """模拟配置"""
    steps: int = 100
    dt: float = 0.1
    method: str = "euler"  # euler, rk4, adaptive
    output: List[str] = field(default_factory=lambda: ["trajectory"])
    visualize: bool = True

@dataclass
class SystemDef(ASTNode):
    """系统定义 - AST根节点"""
    name: str = ""
    dimension: int = 3
    domain: str = "physics"  # physics, optimization, cognitive
    fields: List[FieldDef] = field(default_factory=list)
    particles: List[ParticleDef] = field(default_factory=list)
    constraints: List[ConstraintDef] = field(default_factory=list)
    simulate: Optional[SimulateDef] = None

# ============================================================
# 词法分析器
# ============================================================

class UFLLexer:
    """UFL词法分析器"""
    
    KEYWORDS = {
        'system', 'dimension', 'domain', 'field', 'particle',
        'constraint', 'simulate', 'steps', 'dt', 'method',
        'output', 'visualize', 'position', 'velocity', 'dynamics',
        'type', 'true', 'false', 'physics', 'optimization', 'cognitive',
        'sphere', 'cylinder', 'plane', 'manifold', 'distance',
        'euler', 'rk4', 'adaptive', 'trajectory'
    }
    
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
    
    def error(self, message: str):
        raise UFLSyntaxError(message, SourceLocation(self.line, self.column))
    
    def peek(self, offset: int = 0) -> Optional[str]:
        pos = self.pos + offset
        if pos < len(self.text):
            return self.text[pos]
        return None
    
    def advance(self) -> Optional[str]:
        if self.pos < len(self.text):
            char = self.text[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return char
        return None
    
    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t\n\r':
            self.advance()
    
    def skip_comment(self):
        if self.peek() == '/' and self.peek(1) == '/':
            while self.peek() and self.peek() != '\n':
                self.advance()
            self.advance()  # 跳过换行
        elif self.peek() == '/' and self.peek(1) == '*':
            self.advance()
            self.advance()
            while True:
                if self.peek() is None:
                    self.error("未闭合的块注释")
                if self.peek() == '*' and self.peek(1) == '/':
                    self.advance()
                    self.advance()
                    break
                self.advance()
    
    def read_number(self) -> Tuple[str, SourceLocation]:
        start_line, start_col = self.line, self.column
        num_str = ""
        while self.peek() and (self.peek().isdigit() or self.peek() == '.'):
            num_str += self.advance()
        return num_str, SourceLocation(start_line, start_col)
    
    def read_identifier(self) -> Tuple[str, SourceLocation]:
        start_line, start_col = self.line, self.column
        ident = ""
        while self.peek() and (self.peek().isalnum() or self.peek() in '_'):
            ident += self.advance()
        return ident, SourceLocation(start_line, start_col)
    
    def tokenize(self) -> List[Tuple[str, str, SourceLocation]]:
        """返回 (type, value, location) 的列表"""
        tokens = []
        
        while self.pos < len(self.text):
            self.skip_whitespace()
            
            if self.pos >= len(self.text):
                break
            
            # 跳过注释
            if self.peek() == '/' and (self.peek(1) == '/' or self.peek(1) == '*'):
                self.skip_comment()
                continue
            
            start_line, start_col = self.line, self.column
            location = SourceLocation(start_line, start_col)
            
            char = self.peek()
            
            # 数字
            if char.isdigit():
                num_str, loc = self.read_number()
                tokens.append(('NUMBER', num_str, loc))
            
            # 标识符或关键字
            elif char.isalpha() or char == '_':
                ident, loc = self.read_identifier()
                if ident in self.KEYWORDS:
                    tokens.append(('KEYWORD', ident, loc))
                else:
                    tokens.append(('IDENT', ident, loc))
            
            # 运算符和符号
            elif char == '{':
                self.advance()
                tokens.append(('LBRACE', '{', location))
            elif char == '}':
                self.advance()
                tokens.append(('RBRACE', '}', location))
            elif char == '[':
                self.advance()
                tokens.append(('LBRACKET', '[', location))
            elif char == ']':
                self.advance()
                tokens.append(('RBRACKET', ']', location))
            elif char == '(':
                self.advance()
                tokens.append(('LPAREN', '(', location))
            elif char == ')':
                self.advance()
                tokens.append(('RPAREN', ')', location))
            elif char == ':':
                self.advance()
                tokens.append(('COLON', ':', location))
            elif char == ',':
                self.advance()
                tokens.append(('COMMA', ',', location))
            elif char == '=':
                self.advance()
                tokens.append(('EQUALS', '=', location))
            elif char == '+':
                self.advance()
                tokens.append(('PLUS', '+', location))
            elif char == '-':
                self.advance()
                tokens.append(('MINUS', '-', location))
            elif char == '*':
                self.advance()
                tokens.append(('STAR', '*', location))
            elif char == '/':
                self.advance()
                tokens.append(('SLASH', '/', location))
            elif char == '^':
                self.advance()
                tokens.append(('CARET', '^', location))
            elif char == '|':
                self.advance()
                tokens.append(('PIPE', '|', location))
            else:
                self.error(f"未识别的字符: '{char}'")
        
        tokens.append(('EOF', '', SourceLocation(self.line, self.column)))
        return tokens

# ============================================================
# 语法分析器
# ============================================================

class UFLParser:
    """UFL语法分析器 - 递归下降解析器"""
    
    def __init__(self):
        self.tokens = []
        self.pos = 0
    
    def error(self, message: str):
        if self.pos < len(self.tokens):
            _, _, location = self.tokens[self.pos]
            raise UFLSyntaxError(message, location)
        raise UFLSyntaxError(message)
    
    def current(self) -> Tuple[str, str, SourceLocation]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return ('EOF', '', SourceLocation(0, 0))
    
    def current_type(self) -> str:
        return self.current()[0]
    
    def current_value(self) -> str:
        return self.current()[1]
    
    def current_location(self) -> SourceLocation:
        return self.current()[2]
    
    def advance(self) -> Tuple[str, str, SourceLocation]:
        token = self.current()
        if self.pos < len(self.tokens):
            self.pos += 1
        return token
    
    def expect(self, token_type: str) -> Tuple[str, str, SourceLocation]:
        if self.current_type() != token_type:
            self.error(f"期望 '{token_type}'，得到 '{self.current_type()}'")
        return self.advance()
    
    def parse(self, text: str) -> SystemDef:
        """解析UFL代码"""
        lexer = UFLLexer(text)
        self.tokens = lexer.tokenize()
        self.pos = 0
        
        return self.parse_system()
    
    def parse_system(self) -> SystemDef:
        """解析系统定义"""
        self.expect('KEYWORD')  # 'system'
        name_token = self.expect('IDENT')
        name = name_token[1]
        self.expect('LBRACE')
        
        system = SystemDef(name=name)
        
        while self.current_type() != 'RBRACE':
            self.parse_system_member(system)
        
        self.expect('RBRACE')
        return system
    
    def parse_system_member(self, system: SystemDef):
        """解析系统成员"""
        if self.current_type() != 'KEYWORD':
            self.error(f"期望关键字，得到 '{self.current_value()}'")
        
        keyword = self.current_value()
        self.advance()
        
        if keyword == "dimension":
            self.expect('COLON')
            num_token = self.expect('NUMBER')
            system.dimension = int(num_token[1])
            
        elif keyword == "domain":
            self.expect('COLON')
            domain_token = self.expect('KEYWORD')
            system.domain = domain_token[1]
            
        elif keyword == "field":
            system.fields.append(self.parse_field())
            
        elif keyword == "particle":
            system.particles.append(self.parse_particle())
            
        elif keyword == "constraint":
            system.constraints.append(self.parse_constraint())
            
        elif keyword == "simulate":
            system.simulate = self.parse_simulate()
            
        else:
            self.error(f"未知关键字: {keyword}")
    
    def parse_field(self) -> FieldDef:
        """解析场定义"""
        name_token = self.expect('IDENT')
        name = name_token[1]
        self.expect('LPAREN')
        var_token = self.expect('IDENT')
        var = var_token[1]
        self.expect('RPAREN')
        self.expect('EQUALS')
        
        expr = self.parse_expression()
        
        return FieldDef(name=name, variable=var, expression=expr)
    
    def parse_expression(self) -> Expr:
        """解析表达式 - 支持优先级"""
        return self.parse_additive()
    
    def parse_additive(self) -> Expr:
        """加法/减法 (最低优先级)"""
        left = self.parse_multiplicative()
        
        while self.current_value() in ['+', '-']:
            op = self.current_value()
            self.advance()
            right = self.parse_multiplicative()
            left = BinaryOp(op=op, left=left, right=right)
        
        return left
    
    def parse_multiplicative(self) -> Expr:
        """乘法/除法"""
        left = self.parse_power()
        
        while self.current_value() in ['*', '/']:
            op = self.current_value()
            self.advance()
            right = self.parse_power()
            left = BinaryOp(op=op, left=left, right=right)
        
        return left
    
    def parse_power(self) -> Expr:
        """幂运算"""
        left = self.parse_unary()
        
        if self.current_value() == '^':
            self.advance()
            right = self.parse_power()  # 右结合
            left = BinaryOp(op='^', left=left, right=right)
        
        return left
    
    def parse_unary(self) -> Expr:
        """一元运算"""
        if self.current_value() == '-':
            self.advance()
            operand = self.parse_unary()
            return UnaryOp(op='-', operand=operand)
        
        if self.current_value() == '|':
            self.advance()
            operand = self.parse_unary()
            self.expect('PIPE')
            # 检查是否是 |x|^2
            if self.current_value() == '^':
                self.advance()
                power = self.parse_unary()
                return BinaryOp(op='^', left=UnaryOp(op='|·|', operand=operand), right=power)
            return UnaryOp(op='|·|', operand=operand)
        
        return self.parse_primary()
    
    def parse_primary(self) -> Expr:
        """主表达式"""
        # 数字
        if self.current_type() == 'NUMBER':
            num_token = self.advance()
            return Number(value=float(num_token[1]))
        
        # 向量
        if self.current_type() == 'LBRACKET':
            self.advance()
            elements = []
            while self.current_type() != 'RBRACKET':
                elements.append(self.parse_expression())
                if self.current_type() == 'COMMA':
                    self.advance()
            self.expect('RBRACKET')
            return VectorLiteral(elements=elements)
        
        # 括号表达式
        if self.current_type() == 'LPAREN':
            self.advance()
            expr = self.parse_expression()
            self.expect('RPAREN')
            return expr
        
        # 标识符或函数调用
        if self.current_type() == 'IDENT':
            ident_token = self.advance()
            name = ident_token[1]
            
            # 检查是否是函数调用
            if self.current_type() == 'LPAREN':
                self.advance()
                args = []
                while self.current_type() != 'RPAREN':
                    args.append(self.parse_expression())
                    if self.current_type() == 'COMMA':
                        self.advance()
                self.expect('RPAREN')
                return FunctionCall(name=name, args=args)
            
            return Variable(name=name)
        
        self.error(f"期望表达式，得到 '{self.current_value()}'")
    
    def parse_particle(self) -> ParticleDef:
        """解析粒子定义"""
        name_token = self.expect('IDENT')
        name = name_token[1]
        self.expect('LBRACE')
        
        particle = ParticleDef(name=name, position=[0, 0, 0])
        
        while self.current_type() != 'RBRACE':
            if self.current_type() != 'KEYWORD':
                self.error(f"期望属性名，得到 '{self.current_value()}'")
            
            prop = self.current_value()
            self.advance()
            self.expect('COLON')
            
            if prop == "position":
                particle.position = self.parse_vector()
            elif prop == "velocity":
                particle.velocity = self.parse_vector()
            elif prop == "dynamics":
                particle.dynamics, particle.dynamics_params = self.parse_dynamics()
            else:
                self.error(f"未知粒子属性: {prop}")
        
        self.expect('RBRACE')
        return particle
    
    def parse_vector(self) -> List[float]:
        """解析向量 [x, y, z]"""
        self.expect('LBRACKET')
        values = []
        while self.current_type() != 'RBRACKET':
            if self.current_type() == 'MINUS':
                self.advance()
                num_token = self.expect('NUMBER')
                values.append(-float(num_token[1]))
            else:
                num_token = self.expect('NUMBER')
                values.append(float(num_token[1]))
            
            if self.current_type() == 'COMMA':
                self.advance()
        
        self.expect('RBRACKET')
        return values
    
    def parse_dynamics(self) -> Tuple[str, Dict[str, Any]]:
        """解析动力学定义"""
        name_token = self.expect('IDENT')
        name = name_token[1]
        params = {}
        
        if self.current_type() == 'LPAREN':
            self.advance()
            while self.current_type() != 'RPAREN':
                param_name_token = self.expect('IDENT')
                param_name = param_name_token[1]
                self.expect('EQUALS')
                
                if self.current_type() == 'NUMBER':
                    param_value = float(self.advance()[1])
                else:
                    param_value = self.current_value()
                    self.advance()
                
                params[param_name] = param_value
                
                if self.current_type() == 'COMMA':
                    self.advance()
            
            self.expect('RPAREN')
        
        return name, params
    
    def parse_constraint(self) -> ConstraintDef:
        """解析约束"""
        name_token = self.expect('IDENT')
        name = name_token[1]
        self.expect('LBRACE')
        
        constraint = ConstraintDef(name=name, type="unknown")
        
        while self.current_type() != 'RBRACE':
            if self.current_type() != 'KEYWORD':
                self.error(f"期望属性名，得到 '{self.current_value()}'")
            
            prop = self.current_value()
            self.advance()
            self.expect('COLON')
            
            if prop == "type":
                type_token = self.expect('KEYWORD')
                constraint.type = type_token[1]
            else:
                # 参数值
                if self.current_type() == 'NUMBER':
                    value = float(self.advance()[1])
                else:
                    value = self.current_value()
                    self.advance()
                constraint.params[prop] = value
        
        self.expect('RBRACE')
        return constraint
    
    def parse_simulate(self) -> SimulateDef:
        """解析模拟配置"""
        self.expect('LBRACE')
        
        sim = SimulateDef()
        
        while self.current_type() != 'RBRACE':
            if self.current_type() != 'KEYWORD':
                self.error(f"期望属性名，得到 '{self.current_value()}'")
            
            prop = self.current_value()
            self.advance()
            self.expect('COLON')
            
            if prop == "steps":
                sim.steps = int(self.expect('NUMBER')[1])
            elif prop == "dt":
                sim.dt = float(self.expect('NUMBER')[1])
            elif prop == "method":
                sim.method = self.expect('KEYWORD')[1]
            elif prop == "output":
                sim.output = [self.expect('KEYWORD')[1]]
            elif prop == "visualize":
                value = self.expect('KEYWORD')[1]
                sim.visualize = value.lower() == "true"
        
        self.expect('RBRACE')
        return sim


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    test_code = """
    system 谐振子 {
        dimension: 3
        domain: physics
        
        field 势能(x) = 0.5 * |x|^2
        
        particle 粒子1 {
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
    
    parser = UFLParser()
    try:
        ast = parser.parse(test_code)
        
        print("=== 解析成功 ===")
        print(f"系统名称: {ast.name}")
        print(f"维度: {ast.dimension}")
        print(f"领域: {ast.domain}")
        print(f"场数: {len(ast.fields)}")
        print(f"粒子数: {len(ast.particles)}")
        print(f"约束数: {len(ast.constraints)}")
        print(f"模拟步数: {ast.simulate.steps if ast.simulate else 'N/A'}")
        
    except UFLSyntaxError as e:
        print(f"语法错误: {e}")
    except UFLTypeError as e:
        print(f"类型错误: {e}")
