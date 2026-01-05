# 贡献指南

感谢你对 UFL 项目的兴趣！我们欢迎所有形式的贡献。

## 📋 贡献方式

### 1. 报告 Bug
- 在 GitHub Issues 中创建新 Issue
- 清楚地描述问题
- 提供复现步骤
- 附加相关代码或日志

### 2. 提出功能建议
- 在 GitHub Discussions 中讨论
- 或在 Issues 中标记为 "enhancement"
- 清楚地说明为什么需要这个功能

### 3. 提交代码
- Fork 项目
- 创建新分支：`git checkout -b feature/your-feature`
- 提交更改：`git commit -m "Add your feature"`
- 推送到分支：`git push origin feature/your-feature`
- 创建 Pull Request

### 4. 改进文档
- 修复拼写错误
- 改进示例
- 添加新的教程
- 翻译文档

## 🎯 贡献指南

### 代码风格
- 遵循 PEP 8
- 添加类型注解
- 编写清晰的注释
- 保持代码简洁

### 测试
- 为新功能添加单元测试
- 确保所有测试通过
- 目标：保持 >95% 的测试覆盖率

### 文档
- 更新相关文档
- 添加代码示例
- 更新 API 参考

## 📝 提交信息格式

```
类型: 简短描述

详细描述（可选）

相关 Issue: #123
```

**类型**：
- `feat`: 新功能
- `fix`: 修复 Bug
- `docs`: 文档更新
- `style`: 代码风格
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 其他更改

## 🚀 开发流程

1. Fork 项目
2. 克隆你的 Fork：`git clone https://github.com/your-username/ufl-unified-field-language.git`
3. 添加上游远程：`git remote add upstream https://github.com/liugongshan88-coder/ufl-unified-field-language.git`
4. 创建分支：`git checkout -b feature/your-feature`
5. 进行更改
6. 运行测试：`python3 test_ufl.py`
7. 提交更改：`git commit -m "feat: add your feature"`
8. 推送到你的 Fork：`git push origin feature/your-feature`
9. 创建 Pull Request

## 📊 开发环境设置

```bash
# 克隆项目
git clone https://github.com/liugongshan88-coder/ufl-unified-field-language.git
cd ufl-unified-field-language

# 运行快速开始
python3 quickstart.py

# 运行所有测试
python3 test_ufl.py

# 查看文档
cat README.md
```

## 🎓 学习资源

- [README.md](README.md) - 项目概览
- [GUIDE.md](GUIDE.md) - 完整使用指南
- [UFL_Mathematical_Foundations.md](UFL_Mathematical_Foundations.md) - 数学基础

## 💡 贡献想法

### 短期（容易）
- 修复拼写错误
- 改进文档
- 添加示例
- 优化性能

### 中期（中等）
- 添加新的约束类型
- 实现新的求解器
- 改进错误处理
- 添加新的应用示例

### 长期（困难）
- GPU 加速
- 并行化支持
- 可视化工具
- Web 界面
- 强化学习集成

## ❓ 有问题？

- 查看 [GUIDE.md](GUIDE.md)
- 在 GitHub Discussions 中提问
- 创建 Issue

## 📜 许可证

通过提交贡献，你同意你的贡献将在 MIT 许可证下发布。

---

感谢你的贡献！🎉
