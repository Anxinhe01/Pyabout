import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 读取数据集
data = pd.read_csv('wdbc.data', header=None)

# 设置特征列和目标变量列
X = data.iloc[:, 2:].values
y = (data.iloc[:, 1] == 'M').astype(int)   # 将目标变量转换为二元数值：Malignant（恶性）为1，Benign（良性）为0

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建逻辑回归模型
logistic_reg = LogisticRegression()

# 拟合模型到训练集
logistic_reg.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = logistic_reg.predict(X_test)

# 评估模型性能
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 打印模型性能指标
print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# 获取逻辑回归模型的特征系数
coefficients = logistic_reg.coef_[0]

# 特征名称
feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# 创建一个DataFrame来存储特征名称和对应的系数
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# 根据系数的绝对值对特征进行排序
feature_importance = feature_importance.reindex(feature_importance.Coefficient.abs().sort_values(ascending=False).index)

# 使用水平条形图展示特征重要性
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
plt.xlabel('Coefficient Value')
plt.title('Feature Importance based on Logistic Regression Coefficients')
plt.gca().invert_yaxis()
plt.show()

# 绘制混淆矩阵图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# 显示数据集统计信息
print("Dataset statistics:\n", data.describe())

# 热力相关矩阵
corr = data.iloc[:, 2:].astype(float).corr()
plt.figure(figsize=(25, 20))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()
