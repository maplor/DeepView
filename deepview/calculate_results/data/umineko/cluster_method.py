import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score, silhouette_score
from umap import UMAP
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture


from deepview.calculate_results.models.utils import (
    label_colors,
    majority_value,
    labeldict_findstr,
labeldict_findstr_omizu,
labeldict_findstr_turtle,
)

def kmeans_best_numCenters_elbow(X, init_num_centers=2, max_num_centers=10):
    # 计算不同聚类数下的 SSE
    sse = []
    k_values = range(init_num_centers, max_num_centers)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # 返回最佳聚类数
    # 计算每两个点之间的差值
    diff = np.diff(sse)
    # 计算每两个差值之间的差值
    second_diff = np.diff(diff)
    # 找到第二个差值的最小值的位置
    elbow_point = np.argmin(second_diff) + 1  # 加1是因为我们计算了第二个差值
    print(f"Elbow point: {elbow_point + init_num_centers}")

    # # 绘制肘部法则图
    # plt.figure(figsize=(8, 6))
    # plt.plot(k_values, sse, marker='o')
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('SSE')
    # plt.title('Elbow Method for Optimal k')
    # plt.xticks(k_values)
    # plt.grid()
    # plt.show()
    return elbow_point + init_num_centers


def kmeans_best_numCenters_silhoutte(X, iteration, init_num_centers=2, max_num_centers=10):
    silhouette_scores = []
    k_values = range(init_num_centers, max_num_centers)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))

    # 找到最佳聚类数
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"The optimal number of clusters is: {optimal_k}")

    # # 绘制轮廓系数图
    # if (iteration == 1) or (iteration == 10) or (iteration == 20):
    #     plt.figure(figsize=(8, 6))
    #     plt.plot(k_values, silhouette_scores, marker='o')
    #     plt.xlabel('Number of clusters (k)')
    #     plt.ylabel('Silhouette Score')
    #     plt.title('Silhouette Score for Optimal k')
    #     plt.xticks(k_values)
    #     plt.grid()
    #     plt.title(str(iteration))
    #     plt.show()
    return optimal_k

def plot_func_new_label_marker(X_labeled_list, X_unlabeled_list,
                               X_labeled_pre_list, y_labeled_list,
                                 y_unlabeled_list, y_labeled_pre_list,
                               name):
    label_colors = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'orange',
        4: 'purple',
        5: 'brown',
        6: 'pink',
        7: 'gray',
        8: 'cyan',
        9: 'magenta'
    }
    X_labeled = np.concatenate(X_labeled_list)
    X_unlabeled = np.concatenate(X_unlabeled_list)
    X_labeled_pre = np.concatenate(X_labeled_pre_list)

    y_labeled = np.concatenate(y_labeled_list)
    y_labeled = majority_value(y_labeled)
    # y_labeled_str = [labeldict_findstr[i] for i in y_labeled]

    y_unlabeled = np.concatenate(y_unlabeled_list)
    y_unlabeled = majority_value(y_unlabeled)
    # y_unlabeled_str = [labeldict_findstr[i] for i in y_unlabeled]

    y_labeled_pre = np.concatenate(y_labeled_pre_list)
    y_labeled_pre = majority_value(y_labeled_pre)
    # y_labeled_pre_str = [labeldict_findstr[i] for i in y_labeled_pre]

    # 使用 UMAP 降维到二维
    umap_model = UMAP(n_components=2, random_state=0)
    X_unlabeled_umap = umap_model.fit_transform(X_unlabeled)
    X_labeled_umap = umap_model.transform(X_labeled)
    X_labeled_pre_umap = umap_model.transform(X_labeled_pre)

    # 绘制结果
    plt.figure(figsize=(12, 8))

    # 绘制未标注数据点
    plt.scatter(X_unlabeled_umap[:, 0], X_unlabeled_umap[:, 1],
                color='lightgray', label='Unlabeled', alpha=0.5)

    # 绘制已标注数据点，根据类别着色
    unique_labels = np.unique(y_unlabeled)

    for label in unique_labels:
        idx = (y_labeled == label)
        plt.scatter(X_labeled_umap[idx, 0],
                    X_labeled_umap[idx, 1],
                    label=f'Previous label {labeldict_findstr[label]}',
                    color=label_colors[label],
                    alpha=0.8,
                    marker='o')

    # 绘制更新后的已标注数据点
    for label in unique_labels:
        idx = (y_labeled_pre == label)
        plt.scatter(X_labeled_pre_umap[idx, 0],
                    X_labeled_pre_umap[idx, 1],
                    label=f'Updated label {labeldict_findstr[label]}',
                    color=label_colors[label],
                    alpha=0.8, marker='x')

    # 添加图例和标签
    plt.title('UMAP of Latent Representation')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend()
    plt.grid()
    plt.savefig(f'newlabel_{name}.png', bbox_inches='tight')
    plt.close()

    return


def plot_func(representation_list, sample_list, label_list, pred_list, iteration, name):
    label_concat = np.concatenate(label_list)
    label_concat_vote = majority_value(label_concat)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)

    umap_3d = UMAP(n_components=2)
    proj_3d_gyro = umap_3d.fit_transform(repre_reshape)

    label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]

    # Create a new column for colors based on the labels
    cus_color = [label_colors[i] for i in label_concat_vote]

    if 1:
    # if (iteration == 1) or (iteration == 10) or (iteration == 20) or (iteration == -1):
        fig_3d = px.scatter(
            proj_3d_gyro, x=0, y=1,
            color=label_concat_vote_str,
            # color=label_concat_vote_str,
            labels={'activity': 'activity'},
            color_discrete_map={'ground_stationary': 'red',
                                'stationary': 'blue',
                                'bathing': 'green',
                                'flying_active': 'orange',
                                'flying_passive': 'purple',
                                'foraging': 'brown',
                                }
        )
        # Reduce marker size for all points
        for trace in fig_3d.data:
            trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)

        # Update transparency for traces where activity is '-2.0'
        fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
        fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
        fig_3d.write_html('figures/%s_activeSupBase_epoch_%s.html' % (name, str(iteration)))

    ############################################
    # plot cluster results
    n_clusters = kmeans_best_numCenters_silhoutte(proj_3d_gyro, iteration,
                                                  init_num_centers=6, max_num_centers=12)
    # Create a KMeans instance
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)

    # Fit the model and get cluster labels
    labels = kmeans.fit_predict(proj_3d_gyro)
    cus_color = [label_colors[i] for i in labels]

    if 1:
    # if (iteration == 1) or (iteration == 10) or (iteration == 20):
        fig_3d = px.scatter(
            proj_3d_gyro, x=0, y=1,
            color=cus_color,
            # color=label_concat_vote_str,
            labels={'color': 'cluster'},
            color_discrete_map={'unknown': 'lightgrey'},
        )
        # Reduce marker size for all points
        for trace in fig_3d.data:
            trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)

        # Update transparency for traces where activity is '-2.0'
        fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
        fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
        fig_3d.write_html('figures/%s_activeSupBase_epoch_%s_cluster.html' % (name, str(iteration)))

    # ##############################
    # # calculate score
    # # 假设 y_true 是真实标签，labels 是聚类标签
    # y_true = label_concat_vote_str  # 真实标签
    # labels = labels  # 聚类标签
    #
    # # 计算调整兰德指数
    # ari = adjusted_rand_score(y_true, labels)
    # print(f"Adjusted Rand Index: {ari:.4f}")
    #
    # # 计算归一化互信息
    # nmi = normalized_mutual_info_score(y_true, labels)
    # print(f"Normalized Mutual Information: {nmi:.4f}")
    #
    # # 计算Fowlkes-Mallows指数
    # fmi = fowlkes_mallows_score(y_true, labels)
    # print(f"Fowlkes-Mallows Index: {fmi:.4f}")
    #
    # silhouette = silhouette_score(proj_3d_gyro, labels)
    # print(f"Silhouette Score: {silhouette:.4f}")
    # return ari, nmi, fmi, silhouette
    return 0,0,0,0


def kmeans_clustering(data, min_clusters=6):
    max_clusters = 2 * min_clusters
    best_cluster_labels, best_cluster_c, best_score = 0, 0, 0
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        cluster_centers = kmeans.cluster_centers_
        score = silhouette_score(data, cluster_labels)

        if score > best_score:
            best_score = score
            best_cluster_c = cluster_centers
            best_cluster_labels = cluster_labels
    return {'cluster_labels': best_cluster_labels,
            'cluster_centers': best_cluster_c,
            'best_sil_score': best_score}

def hierarchical_clustering(data, min_clusters=6):
    max_clusters = 2 * min_clusters
    best_score = -1
    best_labels = None

    # 遍历不同的簇数量
    for n_clusters in range(min_clusters, max_clusters + 1):
        # 执行层次聚类
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = hierarchical.fit_predict(data)

        # 计算轮廓系数
        score = silhouette_score(data, cluster_labels)

        # 如果当前轮廓系数更高，则更新最佳结果
        if score > best_score:
            best_score = score
            best_labels = cluster_labels

    return {'cluster_labels': best_labels,
            'cluster_centers': None,  # 层次聚类没有显式的簇中心
            'best_sil_score': best_score}

def dbscan_clustering(data, min_clusters = 6):
    best_score = -1
    min_eps = 0.1
    max_eps = 1.0
    min_samples_range = (5, 20)
    best_cluster_labels = 0
    max_clusters = 2 * min_clusters
    for eps in np.arange(min_eps, max_eps, 0.1):
        for min_samples in range(min_samples_range[0], min_samples_range[1] + 1):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = dbscan.fit_predict(data)

            # DBSCAN 可能会有 -1 类别（表示噪声），这时我们排除噪声
            if (len(set(cluster_labels)) > (min_clusters + 1)) and \
                (len(set(cluster_labels)) < (max_clusters + 1)):  # 有有效的聚类标签
                score = silhouette_score(data, cluster_labels)
                if score > best_score:
                    best_score = score
                    best_cluster_labels = cluster_labels
    return {'cluster_labels': best_cluster_labels,
            'cluster_centers': None,
            'best_sil_score': best_score}

def gmm_clustering(data, min_clusters=6):
    max_clusters = 2 * min_clusters
    best_score = -1
    best_cluster_labels = 0
    best_cluster_centers = 0
    for n_components in range(min_clusters, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        cluster_labels = gmm.fit_predict(data)
        cluster_centers = gmm.means_
        score = silhouette_score(data, cluster_labels)

        if score > best_score:
            best_score = score
            best_cluster_labels = cluster_labels
            best_cluster_centers = cluster_centers
    return {'cluster_labels': best_cluster_labels,
            'cluster_centers': best_cluster_centers,
            'best_sil_score': best_score}

def select_cluster_method(data, min_clusters=6):
    kmeans_result = kmeans_clustering(data, min_clusters)
    hie_result = hierarchical_clustering(data, min_clusters)
    dbscan_result = dbscan_clustering(data, min_clusters)
    gmm_result = gmm_clustering(data, min_clusters)

    results = {'kmeans_result':kmeans_result,
               'hierarchical_result':hie_result,
               'dbscan_result':dbscan_result,
               'gmm_result':gmm_result}
    best_score = 0
    cluster_name = ''
    best_result = None
    # 遍历字典并比较best_sil_score
    for name, result in results.items():
        if result['best_sil_score'] > best_score:
            best_score = result['best_sil_score']
            best_result = result
            cluster_name = name
    return cluster_name, best_result

def cal_cluster_score(data, y_true, labels):
    # 假设 y_true 是真实标签，labels 是聚类标签

    # 计算调整兰德指数
    ari = adjusted_rand_score(y_true, labels)
    print(f"Adjusted Rand Index: {ari:.4f}")

    # 计算归一化互信息
    nmi = normalized_mutual_info_score(y_true, labels)
    print(f"Normalized Mutual Information: {nmi:.4f}")

    # 计算Fowlkes-Mallows指数
    fmi = fowlkes_mallows_score(y_true, labels)
    print(f"Fowlkes-Mallows Index: {fmi:.4f}")

    silhouette = silhouette_score(data, labels)
    print(f"Silhouette Score: {silhouette:.4f}")
    return ari, nmi, fmi, silhouette

def plot_scatter_turtle(data_umap, label_str, iteration, name):
    fig_2d = px.scatter(
        data_umap, x=0, y=1,
        color=label_str,
        labels={'activity': 'activity'},
        color_discrete_map={'surface_behavior': label_colors[0],
                            'social_selfdirected': label_colors[1],
                            'rest_passive': label_colors[2],
                            'locomotion': label_colors[3],
                            'feeding_related': label_colors[4],
                            'exploration_environment': label_colors[5],
                            }
    )
    # Reduce marker size for all points
    for trace in fig_2d.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
    # 更新 x 和 y 轴的标题
    fig_2d.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Color: Activity" # None  # 取消图例标题
    )
    # Update transparency for traces where activity is '-2.0'
    fig_2d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_2d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_2d.write_html(r'D:\code\DeepView\deepview\calculate_results\data\turtle\figures\%s_activeSup_epoch_%s.html' % (name, str(iteration)))

    return


def plot_scatter_umi(data_umap, label_str, iteration, name):
    fig_2d = px.scatter(
        data_umap, x=0, y=1,
        color=label_str,
        labels={'activity': 'activity'},
        color_discrete_map={'ground_stationary': label_colors[0],
                            'stationary': label_colors[1],
                            'bathing': label_colors[2],
                            'flying_active': label_colors[3],
                            'flying_passive': label_colors[4],
                            'foraging': label_colors[5],
                            }
    )
    # Reduce marker size for all points
    for trace in fig_2d.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
    # 更新 x 和 y 轴的标题
    fig_2d.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Color: Activity" # None  # 取消图例标题
    )
    # Update transparency for traces where activity is '-2.0'
    fig_2d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_2d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_2d.write_html(r'D:\code\DeepView\deepview\calculate_results\data\umineko\figures\%s_activeSup_epoch_%s.html' % (name, str(iteration)))

    return


def plot_scatter_omizu(data_umap, label_str, iteration, name):
    fig_2d = px.scatter(
        data_umap, x=0, y=1,
        color=label_str,
        labels={'activity': 'activity'},
        color_discrete_map={'stationary': label_colors[0],
                            'bathing': label_colors[1],
                            'flying': label_colors[2],
                            'foraging': label_colors[3],
                            }
    )
    # Reduce marker size for all points
    for trace in fig_2d.data:
        trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
    # 更新 x 和 y 轴的标题
    fig_2d.update_layout(
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        legend_title="Color: Activity" # None  # 取消图例标题
    )
    # Update transparency for traces where activity is '-2.0'
    fig_2d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    fig_2d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    fig_2d.write_html(r'D:\code\DeepView\deepview\calculate_results\data\omizunagidori\figures\%s_activeSup_epoch_%s_omizu.html' % (name, str(iteration)))

    return


def vis_scatter_label_2d(representation_list, label_list,
                         iteration, name, plot_flag = False, is_omizu=False):
    label_concat = np.concatenate(label_list)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)

    umap_2d = UMAP(n_components=2)
    data_umap = umap_2d.fit_transform(repre_reshape)

    label_concat_vote = majority_value(label_concat)
    if 'omizu' in is_omizu:
        label_concat_vote_str = [labeldict_findstr_omizu[i] for i in label_concat_vote]
    elif is_omizu == 'turtle':
        label_concat_vote_str = [labeldict_findstr_turtle[i] for i in label_concat_vote]
    elif is_omizu == 'umineko':
        label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]
    else:
        print('no label to string dictionary')
    if plot_flag:
        # plot ground truth labels
        if 'omizu' in is_omizu:
            plot_scatter_omizu(data_umap, label_concat_vote_str, iteration, name + '_gt')
        elif 'umineko' == is_omizu:
            plot_scatter_umi(data_umap, label_concat_vote_str, iteration, name+'_gt')
        elif is_omizu == 'turtle':
            plot_scatter_turtle(data_umap, label_concat_vote_str, iteration, name+'_gt')
        else:
            print('no plot for this dataset')

    ############################################
    # # plot cluster labels
    # # select best cluster strategy
    # cluster_name, results = select_cluster_method(data_umap,
    #                                               min_clusters=6)
    # ## plot
    # if plot_flag:
    #     print(f"Best cluster method: {cluster_name}")
    #     plot_scatter(data_umap, results['cluster_labels'], iteration, name+'_cluster', is_omizu=is_omizu)

    ############################################
    # y_true = label_concat_vote_str  # 真实标签
    # labels = results['cluster_labels']  # 聚类标签
    # ari, nmi, fmi, silhouette = cal_cluster_score(data_umap, y_true, labels)
    # return ari, nmi, fmi, silhouette
    return 0,0,0,0