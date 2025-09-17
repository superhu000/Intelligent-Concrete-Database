import warnings
import io
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from kneed import KneeLocator  # 通过kneed库自动检测手肘点
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'

sns.set(font='SimHei', context="notebook", color_codes=True)

plt.style.use('bmh')

# %matplotlib inline
pd.set_option('display.max_columns', None)


# feature_col = ['C-CaO', 'C-SiO2', 'C-Al2O3', 'C-MgO', 'C-Fe2O3', '水泥掺量', '水掺量', '水灰比', '孔隙率']
# target_col = ['28d抗压强度']


# 由特征列和目标列生成
def divide(feature_col, target_col):
    df_kx = pd.read_excel(r"E:\Study\Project\MaterialGene\Coding\backend\concrete\data\output.xlsx")
    df_kx1 = df_kx.dropna(axis=0, how='any', subset=('C-CaO', 'C-SiO2', 'C-Al2O3', 'C-MgO', 'C-Fe2O3', "28d抗压强度"))
    df_copy = df_kx1.copy()

    features = df_copy[feature_col]  # 特征列
    targets = df_copy[target_col]  # 目标列

    seed = 16
    train_x, test_x, train_y, test_y = train_test_split(features, targets, test_size=0.3, random_state=seed)
    train_y = train_y[target_col[0]]  # 将train_y和test_y中的目标列提取出来，分别赋值给train_y和test_y
    test_y = test_y[target_col[0]]

    return train_x, test_x, train_y, test_y


# 定义了一个名为evaluate_model的函数，它接收8个参数：特征列和目标列、训练集的特征和目标、测试集的特征和目标、模型的类型、模型算法
# 以及一个可选参数plot，默认值为True。
def evaluate(feature_col, target_col, train_x, train_y, test_x, test_y, types, alg, plot=True):
    alg.fit(train_x, train_y)
    pal = sns.color_palette("hls", 10)
    # print(f"Score: {alg.score(test_x, test_y)}")

    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        try:
            if types == "Coefs":
                print(f"Intercept: {alg.intercept_}")
                try:
                    coef = pd.DataFrame({"coefs": alg.coef_, "col": feature_col})
                except:
                    coefs = pd.DataFrame({"coefs": alg.coef_[0], "col": feature_col})
                sns.barplot(x="col", y="coefs", data=coefs, ax=axes[1], palette=pal)
                axes[1].set_title("性能基因量化分析")
            else:
                features = pd.DataFrame({"features": alg.feature_importances_, "col": feature_col})
                sns.barplot(x="col", y="features", data=features, ax=axes[1], palette=pal)
                axes[1].set_title("性能基因量化分析")
        except:
            pass
    else:
        plt.figure(figsize=(20, 5))
        axes = [None]

    pred = alg.predict(test_x)
    rmse = np.sqrt(metrics.mean_squared_error(test_y, pred))
    mae = mean_absolute_error(test_y, pred)
    mse = mean_squared_error(test_y, pred)
    r2 = r2_score(test_y, pred)
    print(r2)

    p = pd.DataFrame(pred, columns=[0])
    p["Type"] = "Predictions"
    p["n"] = list(range(p.shape[0]))
    t = test_y.copy()
    t = t.reset_index().set_index("index")
    t.columns = [0]
    t["Type"] = "Actual"
    t = t[t[0] != "Actual"]
    t["n"] = list(range(p.shape[0]))
    x = pd.concat([p, t], axis=0).reset_index()
    sns.lineplot(x="n", y=0, hue="Type", data=x, markers=["o", "o"], style="Type", ax=axes[0])
    axes[0].set_title("拟合结果分析")  # 为第一个图设置标题

    # 保存图像为 BytesIO 对象
    plt.show()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    # print(img_buffer)

    return rmse, mae, mse, r2, img_buffer


def correlation(data, num_1):
    # 创建 DataFrame
    df = pd.DataFrame(data)

    def fill_space_with_mean(row):
        for col in df.columns:
            if np.isnan(row[col]):  # 检查是否为空值
                mean_value = df[col].mean()  # 计算列的均值
                row[col] = mean_value  # 填充空值

    # 用每列的平均值填充空值
    # df.fillna(df.mean(), inplace=True)
    df.apply(fill_space_with_mean, axis=1)
    print(df)
    df.dropna(axis=1, how='any', inplace=True)

    # 降维前的维度
    original_shape = df.shape
    print(f"降维前的维度: {original_shape}")

    # 使用PCA降维
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df)

    # # PCA降维：选择保留95%方差的维度数，并打印降维前后的维度变化
    # pca = PCA(n_components=0.95)
    # data_pca = pca.fit_transform(data_scaled)
    #
    # print(f"Original data shape: {data_scaled.shape}")
    # print(f"PCA reduced data shape: {data_pca.shape}")

    # 将 PCA 结果转换为 DataFrame
    df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])

    # 降维后的维度
    reduced_shape = df_pca.shape
    print(f"降维后的维度: {reduced_shape}")

    # 聚类 (可选步骤)
    kmeans = KMeans(n_clusters=num_1, random_state=0)
    df_pca['Cluster'] = kmeans.fit_predict(df_pca)

    # 使用 t-SNE 进行可视化
    tsne = TSNE(n_components=2, perplexity=2, n_iter=300, random_state=0)
    tsne_result = tsne.fit_transform(df)

    # 将 t-SNE 结果转换为 DataFrame
    df_tsne = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])

    # 将聚类结果加入 t-SNE 数据中 (如果执行了聚类)
    df_tsne['Cluster'] = df_pca['Cluster']

    # 可视化 PCA 结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_pca, palette='viridis')
    plt.title('PCA降维效果可视化')

    # 可视化 t-SNE 结果
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue='Cluster', data=df_tsne, palette='viridis')
    plt.title('基因数据相似度量可视化')

    plt.show()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    print(img_buffer)

    return original_shape, reduced_shape, img_buffer


def correlation1():
    # 1. 加载数据集并使用10%数据
    california_housing = fetch_california_housing()
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)

    # 使用10%的数据进行分析
    data, _ = train_test_split(data, test_size=0.9, random_state=42)

    # 检查并填补缺失值（本数据集中没有缺失值，但为了演示填补方法）
    data.fillna(data.mean(), inplace=True)

    # 添加目标变量
    data['MedHouseVal'] = california_housing.target[:len(data)]

    # 数据标准化
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data.drop(columns=['MedHouseVal']))

    # 2. PCA降维：选择保留95%方差的维度数，并打印降维前后的维度变化
    pca = PCA(n_components=0.95)
    data_pca = pca.fit_transform(data_scaled)

    print(f"Original data shape: {data_scaled.shape}")
    print(f"PCA reduced data shape: {data_pca.shape}")

    # 3. 使用手肘法确定最佳聚类数并进行K-means聚类
    sse = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data_pca)
        sse.append(kmeans.inertia_)

    # 使用kneed库来检测手肘点，确定最佳聚类数
    kneedle = KneeLocator(k_range, sse, curve="convex", direction="decreasing")
    optimal_k = kneedle.elbow
    print(f"The optimal number of clusters determined by the elbow method is: {optimal_k}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.vlines(optimal_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('SSE')
    plt.grid(True)
    plt.show()

    img_buffer1 = io.BytesIO()
    plt.savefig(img_buffer1, format='png')
    img_buffer1.seek(0)
    plt.clf()

    # 根据手肘法选择最佳k值进行聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    labels = kmeans.fit_predict(data_pca)

    # 4. 聚类可视化和t-SNE可视化
    # 绘制聚类后的散点图
    plt.figure(figsize=(16, 8))

    # 聚类后的散点图
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter, label='Cluster Label')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
    plt.title(f'K-Means Clustering on PCA Reduced Data (k={optimal_k})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend()

    # t-SNE可视化
    tsne = TSNE(n_components=2, random_state=42)
    data_tsne = tsne.fit_transform(data_pca)

    plt.subplot(1, 2, 2)
    scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter, label='Cluster Label')
    plt.title('t-SNE Visualization of PCA Reduced Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.show()

    # 保存图像为 BytesIO 对象
    plt.show()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    return data_scaled.shape, data_pca.shape, img_buffer, img_buffer1
