LendingClub 贷款数据分析项目

项目简介
这是一个针对LendingClub贷款数据的分析项目，旨在通过机器学习模型预测贷款违约风险。

项目结构
```
LendingClub/
├── data/               # 数据目录（本地，不上传）
│   ├── raw/           # 原始数据
│   └── processed/     # 处理后的数据
├── src/               # 源代码
│   ├── data_processing.py    # 数据清洗和处理
│   ├── feature_engineering.py # 特征工程
│   ├── model_training.py     # 模型训练
│   └── visualization.py      # 数据可视化
├── notebooks/         # Jupyter笔记本
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_training.ipynb
├── outputs/           # 输出结果
│   ├── models/        # 训练好的模型（过大，不上传）
│   ├── plots/         # 生成图表
│   └── reports/       # 分析报告
├── requirements.txt   # Python依赖
└── README.md         # 项目说明
```

快速开始

环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

数据准备
由于数据文件较大，请从[[Google Drive](https://drive.google.com/...)](https://www.kaggle.com/datasets/wordsforthewise/lending-club)下载并放入`data/raw/`目录：
- accepted_2007_to_2018Q4.csv.gz

运行流程
1. 数据预处理：`python src/data_processing.py`
2. 特征工程：`python src/quick_eda.py`
3. 模型训练：`python src/model_building.py`
4. 自动化策略决策系统：'python src/strategy_output.py'

## 📊 文件说明

源代码文件
data_processing.py: 数据清洗、缺失值处理、数据类型转换
quick_eda.py: 创建新特征、特征选择、数据标准化
model_building.py: 训练随机森林和XGBoost模型，模型评估
strategy_output.py:从原始数据到可执行策略的端到端转化器


数据文件
原始数据: 来自LendingClub 2007-2018Q4的贷款数据
处理后的数据: 经过清洗和特征工程的数据集

主要发现
发现了影响贷款违约的关键特征：FICO分数、债务收入比、贷款金额
XGBoost模型在测试集上达到92%的准确率
特征重要性分析显示FICO分数是最重要的预测因子

 贡献指南
1. Fork本项目
2. 创建特性分支：`git checkout -b feature/AmazingFeature`
3. 提交更改：`git commit -m 'Add some AmazingFeature'`
4. 推送到分支：`git push origin feature/AmazingFeature`
5. 提交Pull Request

🙏 致谢
- 数据来源：LendingClub
