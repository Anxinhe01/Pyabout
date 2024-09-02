import pandas as pd

# 加载数据
movies_df = pd.read_csv('tmdb_5000_movies.csv')

# 数据清洗
# 假设我们删除票房收入（revenue）为NaN的行
movies_df = movies_df.dropna(subset=['revenue'])

# 数据探索
# 查看票房收入的描述性统计
revenue_stats = movies_df['revenue'].describe()
print(revenue_stats)

# 统计分析
# 计算每种类型电影的平均票房收入
genre_revenue = movies_df.std.split('|').apply(lambda x: [i for i in x if i]).explode().groupby('genres').mean().reset_index()

# 文字分析
analysis = """
票房收入分析：
- 平均票房收入: ${:,.2f}
- 中位数票房收入: ${:,.2f}
- 最高票房收入: ${:,.2f}
- 最低票房收入: ${:,.2f}
""".format(
    revenue_stats['mean'],
    revenue_stats['50%'],
    revenue_stats['max'],
    revenue_stats['min']
)

# 电影类型分析
genre_analysis = """
电影类型与票房收入分析：
- 平均票房收入最高的电影类型是 '{}'，平均票房收入为 ${:,.2f}。
- 平均票房收入最低的电影类型是 '{}'，平均票房收入为 ${:,.2f}。
""".format(
    genre_revenue['genres'].iloc[0],
    genre_revenue['revenue'].iloc[0],
    genre_revenue['genres'].iloc[-1],
    genre_revenue['revenue'].iloc[-1]
)

# 打印分析结果
print(analysis)
print(genre_analysis)