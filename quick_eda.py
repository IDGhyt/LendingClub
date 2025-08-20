import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建可视化输出目录
os.makedirs('outputs/visualizations', exist_ok=True)

# 加载精简数据
print("加载精简数据...")
try:
    df = pd.read_csv('outputs/data/loan_quick.csv', parse_dates=['issue_d'])

    # 数据质量检查 - 体现数据分析基本功
    print("数据基本信息:")
    print(f"数据集形状: {df.shape}")
    print("\n各列缺失值情况:")
    print(df.isnull().sum().sort_values(ascending=False))

    # 关键指标统计 - 展示业务理解能力
    print("\n关键业务指标:")
    print(f"贷款笔数: {len(df):,}")
    print(f"平均贷款金额: ${df['loan_amnt'].mean():.2f}")
    print(f"平均利率: {df['int_rate'].mean():.2f}%")
    print(f"违约率: {(df['loan_status'] == 'Charged Off').mean() * 100:.2f}%")

except FileNotFoundError:
    print("数据文件未找到，请检查路径")
    # 创建示例数据用于演示
    np.random.seed(42)
    n_samples = 10000
    df = pd.DataFrame({
        'fico_range_low': np.random.randint(600, 800, n_samples),
        'loan_status': np.random.choice(['Fully Paid', 'Charged Off'], n_samples, p=[0.9, 0.1]),
        'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_samples,
                                  p=[0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02]),
        'int_rate': np.random.uniform(5, 25, n_samples),
        'annual_inc': np.random.lognormal(10.5, 0.4, n_samples),
        'loan_amnt': np.random.randint(1000, 35000, n_samples),
        'dti': np.random.uniform(5, 35, n_samples),
        'issue_d': pd.date_range('2010-01-01', periods=n_samples, freq='D')
    })
    print("使用模拟数据进行演示")

# 1. 违约率 vs 信用分 - 展示风险识别能力
plt.figure(figsize=(12, 7))
default_rates = df.groupby('fico_range_low')['loan_status'].apply(
    lambda x: (x == 'Charged Off').mean() * 100  # 转换为百分比
)

# 添加趋势线
z = np.polyfit(default_rates.index, default_rates.values, 2)
p = np.poly1d(z)

plt.plot(default_rates.index, default_rates.values, 'o', alpha=0.7, label='实际违约率')
plt.plot(default_rates.index, p(default_rates.index), "r--", linewidth=2, label='趋势线')

plt.title('FICO分数与违约率关系', fontsize=16, fontweight='bold')
plt.xlabel('FICO分数', fontsize=12)
plt.ylabel('违约率(%)', fontsize=12)

# 添加风险阈值区域
plt.axhspan(0, 5, alpha=0.2, color='green', label='低风险区')
plt.axhspan(5, 10, alpha=0.2, color='yellow', label='中风险区')
plt.axhspan(10, 100, alpha=0.2, color='red', label='高风险区')

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/visualizations/fico_vs_default.png', dpi=300)
plt.close()
print("生成图表: FICO分数与违约率关系 - 展示风险识别能力")

# 2. 不同等级的利率分布 - 展示定价策略理解
plt.figure(figsize=(12, 7))
df['grade'] = df['grade'].astype(str)
valid_grades = sorted([g for g in df['grade'].unique() if g != 'nan'])

# 计算每个等级的违约率
default_by_grade = df.groupby('grade')['loan_status'].apply(
    lambda x: (x == 'Charged Off').mean() * 100
).reindex(valid_grades)

# 创建双轴图
fig, ax1 = plt.subplots(figsize=(12, 7))

# 利率箱线图
sns.boxplot(data=df[df['grade'].isin(valid_grades)],
            x='grade',
            y='int_rate',
            order=valid_grades,
            ax=ax1)
ax1.set_title('不同信用等级的利率与违约率关系', fontsize=16, fontweight='bold')
ax1.set_xlabel('信用等级', fontsize=12)
ax1.set_ylabel('利率(%)', fontsize=12)

# 添加违约率折线图
ax2 = ax1.twinx()
ax2.plot(range(len(valid_grades)), default_by_grade.values,
         'r-o', linewidth=2, markersize=8, label='违约率')
ax2.set_ylabel('违约率(%)', fontsize=12)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('outputs/visualizations/grade_vs_interest.png', dpi=300)
plt.close()
print("生成图表: 不同信用等级的利率分布 - 展示定价策略理解")

# 3. 贷款金额与收入的关系 - 展示客户分层能力
# 添加收入分层
df['income_segment'] = pd.cut(df['annual_inc'],
                              bins=[0, 30000, 60000, 100000, 200000, np.inf],
                              labels=['低收入', '中低收入', '中等收入', '中高收入', '高收入'])

fig = px.scatter(df.sample(n=5000, random_state=42),  # 减少样本量提高性能
                 x='annual_inc',
                 y='loan_amnt',
                 color='grade',
                 opacity=0.6,
                 hover_data=['int_rate', 'dti', 'income_segment'],
                 title='年收入与贷款金额关系（按信用等级分层）',
                 labels={'annual_inc': '年收入($)', 'loan_amnt': '贷款金额($)', 'grade': '信用等级'})
fig.update_xaxes(range=[0, 200000])
fig.update_layout(
    title_font_size=16,
    title_x=0.5,
    font=dict(size=10)
)
fig.write_html('outputs/visualizations/income_vs_loan.html')
print("生成交互图表: 年收入与贷款金额关系 - 展示客户分层能力")

# 4. 新增：时间趋势分析 - 展示业务监控能力
plt.figure(figsize=(14, 8))
monthly_data = df.groupby(pd.Grouper(key='issue_d', freq='M')).agg({
    'loan_amnt': 'mean',
    'int_rate': 'mean',
    'loan_status': lambda x: (x == 'Charged Off').mean() * 100
}).rename(columns={'loan_status': 'default_rate'}).dropna()

# 双轴图展示贷款金额和违约率趋势
fig, ax1 = plt.subplots(figsize=(14, 8))

color = 'tab:blue'
ax1.set_xlabel('时间')
ax1.set_ylabel('平均贷款金额($)', color=color)
ax1.plot(monthly_data.index, monthly_data['loan_amnt'], color=color, linewidth=2.5)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('违约率(%)', color=color)
ax2.plot(monthly_data.index, monthly_data['default_rate'], color=color, linewidth=2.5)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('贷款金额与违约率时间趋势', fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/visualizations/trend_analysis.png', dpi=300)
plt.close()
print("生成图表: 时间趋势分析 - 展示业务监控能力")

# 5. 新增：关键指标异常检测 - 展示监控指标设置能力
print("\n异常检测报告:")
# 简单异常检测 - 识别近期异常波动
recent_data = monthly_data.tail(6)  # 最近6个月

# 计算移动平均值和标准差
mean_loan = monthly_data['loan_amnt'].rolling(window=6).mean().iloc[-1]
std_loan = monthly_data['loan_amnt'].rolling(window=6).std().iloc[-1]

mean_default = monthly_data['default_rate'].rolling(window=6).mean().iloc[-1]
std_default = monthly_data['default_rate'].rolling(window=6).std().iloc[-1]

current_loan = recent_data['loan_amnt'].iloc[-1]
current_default = recent_data['default_rate'].iloc[-1]

# 检测异常（超过2个标准差）
if abs(current_loan - mean_loan) > 2 * std_loan:
    print(
        f"⚠️  贷款金额异常: 当前值${current_loan:.2f}, 预期范围${mean_loan - 2 * std_loan:.2f}-${mean_loan + 2 * std_std_loan:.2f}")

if abs(current_default - mean_default) > 2 * std_default:
    print(
        f"⚠️  违约率异常: 当前值{current_default:.2f}%, 预期范围{mean_default - 2 * std_default:.2f}-{mean_default + 2 * std_default:.2f}%")

if abs(current_loan - mean_loan) <= 2 * std_loan and abs(current_default - mean_default) <= 2 * std_default:
    print("✅ 所有关键指标在正常范围内")

print("\nEDA完成! 查看outputs/visualizations目录获取图表")