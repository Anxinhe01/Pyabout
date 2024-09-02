import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集
iris_data = pd.read_csv("iris.data", header=None)
iris = load_iris()
# 重命名列以提高可读性
iris_data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# 展示数据表前5个数据
print(iris_data.head())

# 检查数据的统计信息
print("数据描述统计:")
print(iris_data.describe())

# 数据加载与准备
X = iris_data.iloc[:, :-1].values  # 获取特征变量
y = iris_data['species'].values  # 目标变量

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 决策树模型构建
model = DecisionTreeClassifier(random_state=42, criterion='entropy')

# 模型训练
model.fit(X_train, y_train)

# 模型评估
y_pred = model.predict(X_test)
print("分类报告:")
print(classification_report(y_test, y_pred))
print("测试集准确率: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

# 可视化混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris_data['species'].unique(), yticklabels=iris_data['species'].unique())
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()



# 特征重要性可视化
feature_importances = pd.DataFrame(model.feature_importances_, index=iris_data.columns[:-1], columns=['importance']).sort_values('importance', ascending=False)
print("特征重要性:")
print(feature_importances)



plt.figure(figsize=(10, 6))
feature_importances.plot(kind='barh', color='lightblue')
plt.title('特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.show()


# 决策树深度与准确率的关系
max_depths = range(1, 11)
train_accuracies = []
test_accuracies = []
for depth in max_depths:
    clf = DecisionTreeClassifier(random_state=42, criterion='entropy', max_depth=depth)
    clf.fit(X_train, y_train)
    train_accuracies.append(clf.score(X_train, y_train))
    test_accuracies.append(clf.score(X_test, y_test))

# 绘制决策树图
plt.figure(figsize=(20, 10))
tree.plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_accuracies, label='训练集准确率')
plt.plot(max_depths, test_accuracies, label='测试集准确率')
plt.xlabel('决策树最大深度')
plt.ylabel('准确率')
plt.title('决策树深度与准确率的关系')
plt.legend()
plt.grid(True)
plt.show()