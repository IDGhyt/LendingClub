import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负piop p号显示问题

# 创建可视化输出目录
os.makedirs('outputs/visualizations', exist_ok=True)

# 加载精简数据
print("加载精简数据...")
df = pd.read_csv('outputs/data/loan_quick.csv', parse_dates=['issue_d'])

# 1. 违约率 vs 信用分
plt.figure(figsize=(10, 6))
default_rates = df.groupby('fico_range_low')['loan_status'].apply(
    lambda x: (x == 'Charged Off').mean()
)
default_rates.plot(title='FICO分数与违约率关系', linewidth=2.5)
plt.xlabel('FICO分数')
plt.ylabel('违约率')
plt.axhline(y=0.05, color='r', linestyle='--', label='5%风险阈值')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/visualizations/fico_vs_default.png', dpi=300)
print("生成图表: FICO分数与违约率关系")

# 2. 不同等级的利率分布
plt.figure(figsize=(10, 6))
# 确保grade列是字符串类型并去除可能的缺失值
df['grade'] = df['grade'].astype(str)
valid_grades = sorted([g for g in df['grade'].unique() if g != 'nan'])
sns.boxplot(data=df[df['grade'].isin(valid_grades)],
            x='grade',
            y='int_rate',
            order=valid_grades)
plt.title('不同信用等级的利率分布')
plt.xlabel('信用等级')
plt.ylabel('利率(%)')
plt.savefig('outputs/visualizations/grade_vs_interest.png', dpi=300)
print("生成图表: 不同信用等级的利率分布")

# 3. 贷款金额与收入的关系
fig = px.scatter(df.sample(n=10000),
                x='annual_inc',
                y='loan_amnt',
                color='grade',
                opacity=0.5,
                hover_data=['int_rate', 'dti'],
                title='年收入与贷款金额关系',
                labels={'annual_inc': '年收入($)', 'loan_amnt': '贷款金额($)'})
fig.update_xaxes(range=[0, 200000])
fig.write_html('outputs/visualizations/income_vs_loan.html')
print("生成交互图表: 年收入与贷款金额关系")

print("EDA完成! 查看outputs/visualizations目录获取图表")