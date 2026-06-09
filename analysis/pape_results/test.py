import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.neighbors import NearestNeighbors


# =========================
# 1. 工具函数
# =========================

def _scale(X):
    """标准化到 0 均值、1 方差。"""
    return StandardScaler().fit_transform(X)


def _cluster_full(X, prefer_hdbscan=True, random_state=42):
    """
    简化版 cluster_full：这里直接用 KMeans(k=4)，对应 4 个 resting 子类。
    你可以在自己的项目里换回原版 _cluster_full。
    """
    km = KMeans(n_clusters=4, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    extra = {
        "auto_k": 4,
        "noise_ratio": 0.0,   # 这里没有 noise 概念，就设 0
    }
    return labels, "kmeans_k4", extra


def _internal_cluster_metrics(X, y_pred):
    """计算 silhouette / DB / CH，若无法定义则返回 NaN。"""
    X = np.asarray(X)
    y_pred = np.asarray(y_pred)
    unique = np.unique(y_pred)
    if unique.size < 2 or X.shape[0] < 5:
        return {"silhouette": np.nan, "db": np.nan, "ch": np.nan}

    try:
        sil = silhouette_score(X, y_pred)
    except Exception:
        sil = np.nan

    try:
        db = davies_bouldin_score(X, y_pred)
    except Exception:
        db = np.nan

    try:
        ch = calinski_harabasz_score(X, y_pred)
    except Exception:
        ch = np.nan

    return {"silhouette": sil, "db": db, "ch": ch}


def _hopkins(X, m=0.1, random_state=42):
    """
    简化版 Hopkins 统计量。
    ~0.5：更像均匀噪声；接近 1：更“可聚类”。
    """
    X = np.asarray(X)
    n, d = X.shape
    m = int(m * n) if isinstance(m, float) else m
    m = max(10, min(m, n - 1))

    rng = np.random.RandomState(random_state)

    # 从真实数据中抽样 m 个点
    idx = rng.choice(np.arange(n), size=m, replace=False)
    X_sample = X[idx]

    # 在边界内随机采样 m 个点
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    X_uniform = rng.uniform(mins, maxs, size=(m, d))

    # 最近邻模型
    nn_X = NearestNeighbors(n_neighbors=2).fit(X)

    # 对 X_sample：取第二近邻（排除自身）
    dist_X, _ = nn_X.kneighbors(X_sample)
    w = dist_X[:, 1].sum()

    # 对 X_uniform：到 X 的最近邻
    dist_U, _ = nn_X.kneighbors(X_uniform, n_neighbors=1)
    u = dist_U[:, 0].sum()

    H = u / (u + w)
    return float(H)


def _safe(v):
    try:
        if v is None:
            return np.nan
        v = float(v)
        if np.isnan(v):
            return np.nan
        return v
    except Exception:
        return np.nan


def _inv_db(db):
    """把 DB 转成 1/DB，让“越大越好”统一起来。"""
    db = _safe(db)
    if db is None or np.isnan(db) or db <= 0:
        return np.nan
    return 1.0 / db


def _scale_01_pos(ch):
    """
    简单的 0-1 归一：v / (v + 1000)，只保证单调性即可。
    真实代码里你可以换成全局 min-max 归一。
    """
    ch = _safe(ch)
    if np.isnan(ch) or ch <= 0:
        return 0.0
    return ch / (ch + 1000.0)


# =========================
# 2. panel A 主函数
# =========================

def panel_A_metrics(X_all, y_coarse, resting_class, sub_rest,
                    out_png="panelA_demo.png",
                    out_csv="panelA_demo.csv",
                    random_state=42):
    """
    只在 resting 子集上做无监督聚类，计算聚类/可聚类性指标。
    输入：
        X_all: dict, {"SupCon": repr_sup, "CEonly": repr_ce}
        y_coarse: 粗粒度标签（len = N）
        resting_class: int，哪一个 coarse label 是 resting
        sub_rest: resting 子集的真子类标签（len = n_rest）
    输出：
        画出条形图，保存 csv，并返回 sup/ce 的原始指标。
    """
    def compute_one(X):
        mask_rest = (y_coarse == resting_class)
        Xr = _scale(X)[mask_rest]

        # 聚类
        y_pred, algo, extra = _cluster_full(
            Xr, prefer_hdbscan=True, random_state=random_state
        )
        intern = _internal_cluster_metrics(Xr, y_pred)
        hop = _hopkins(Xr, m=0.1, random_state=random_state)

        # 与真子类的 NMI/ARI
        nmi = ari = np.nan
        if sub_rest is not None and len(sub_rest) == len(Xr):
            mask_eval = (y_pred != -1)   # 这里没有 noise，一般全 True
            yr_true = np.asarray(sub_rest)[mask_eval]
            yr_pred = y_pred[mask_eval]
            if len(np.unique(yr_true)) >= 2 and len(np.unique(yr_pred)) >= 2:
                nmi = normalized_mutual_info_score(yr_true, yr_pred)
                ari = adjusted_rand_score(yr_true, yr_pred)

        result = {
            "algo": extra.get("auto_k", None),
            "auto_k": extra.get("auto_k", None),
            "noise_ratio": extra.get("noise_ratio", 0.0),
            "silhouette": intern["silhouette"],
            "db": intern["db"],
            "ch": intern["ch"],
            "hopkins": hop,
            "nmi_true": nmi,
            "ari_true": ari
        }
        return result

    sup = compute_one(X_all["SupCon"])
    ce = compute_one(X_all["CEonly"])

    metrics = ["nmi_true", "ari_true", "silhouette", "1/DB", "CH (scaled)", "hopkins"]
    sup_vals = [
        _safe(sup["nmi_true"]), _safe(sup["ari_true"]),
        _safe(sup["silhouette"]),
        _inv_db(sup["db"]),
        _scale_01_pos(sup["ch"]),
        _safe(sup["hopkins"]),
    ]
    ce_vals = [
        _safe(ce["nmi_true"]), _safe(ce["ari_true"]),
        _safe(ce["silhouette"]),
        _inv_db(ce["db"]),
        _scale_01_pos(ce["ch"]),
        _safe(ce["hopkins"]),
    ]

    df = pd.DataFrame({"Metric": metrics,
                       "CE-only": ce_vals,
                       "CE+SupCon": sup_vals})
    df.to_csv(out_csv, index=False)

    # 画 pannel A 条形图
    plt.figure(figsize=(8, 4.8), dpi=150)
    x = np.arange(len(metrics))
    w = 0.35
    plt.bar(x - w/2, df["CE-only"], width=w, label="CE-only")
    plt.bar(x + w/2, df["CE+SupCon"], width=w, label="CE+SupCon")
    plt.xticks(x, metrics, rotation=15, ha="right")
    plt.ylabel("Score (normalized)")
    plt.title("Panel A — A-class clustering quality & auto-K proxies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.show()
    plt.close()

    return {"sup": sup, "ce": ce, "csv": out_csv, "png": out_png, "df": df}


# =========================
# 3. 构造两组 latent + 测试
# =========================

def build_synthetic_latent(random_state=0):
    """
    构造一个 toy 数据：
    - coarse: 0 = resting, 1 = others
    - resting 内有 4 个子类 sub_rest
    - CEonly: 子类重叠严重
    - SupCon: 子类更分离、紧凑
    """
    rng = np.random.RandomState(random_state)

    n_rest = 400   # resting 样本数
    n_other = 100  # 非 resting 样本数
    n_total = n_rest + n_other

    # coarse label: 0 = resting, 1 = others
    y_coarse = np.zeros(n_total, dtype=int)
    y_coarse[n_rest:] = 1
    resting_class = 0

    # resting 子类标签（4 个）
    sub_rest = rng.randint(0, 4, size=n_rest)

    # ---- CE-only latent：更糊 ----
    centers_ce = np.array([
        [0.0, 0.0],
        [2.0, 0.5],
        [0.5, 2.0],
        [2.0, 2.0],
    ])
    X_ce_rest = np.zeros((n_rest, 2))
    for k in range(4):
        mask = (sub_rest == k)
        X_ce_rest[mask] = centers_ce[k] + rng.normal(
            scale=1.0, size=(mask.sum(), 2)
        )

    # ---- SupCon latent：更分离、更紧凑 ----
    centers_sup = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [0.0, 5.0],
        [5.0, 5.0],
    ])
    X_sup_rest = np.zeros((n_rest, 2))
    for k in range(4):
        mask = (sub_rest == k)
        X_sup_rest[mask] = centers_sup[k] + rng.normal(
            scale=0.4, size=(mask.sum(), 2)
        )

    # ---- non-resting 部分随便放远一点 ----
    X_ce = np.zeros((n_total, 2))
    X_ce[:n_rest] = X_ce_rest
    X_ce[n_rest:] = rng.normal(loc=[6.0, 6.0], scale=1.5, size=(n_other, 2))

    X_sup = np.zeros((n_total, 2))
    X_sup[:n_rest] = X_sup_rest
    X_sup[n_rest:] = rng.normal(loc=[10.0, 10.0], scale=1.5, size=(n_other, 2))

    return X_sup, X_ce, y_coarse, resting_class, sub_rest


def plot_latent_resting(X_sup, X_ce, y_coarse, resting_class, sub_rest):
    """画出两组 latent 在 resting 子集上的散点图。"""
    mask_rest = (y_coarse == resting_class)

    plt.figure(figsize=(10, 4), dpi=120)
    plt.suptitle("Synthetic latent spaces: CE-only vs SupCon", y=1.05)

    plt.subplot(1, 2, 1)
    plt.scatter(
        X_ce[mask_rest, 0],
        X_ce[mask_rest, 1],
        c=sub_rest,
        s=15,
    )
    plt.title("CE-only latent (resting subset)")
    plt.xlabel("dim1")
    plt.ylabel("dim2")

    plt.subplot(1, 2, 2)
    plt.scatter(
        X_sup[mask_rest, 0],
        X_sup[mask_rest, 1],
        c=sub_rest,
        s=15,
    )
    plt.title("SupCon latent (resting subset)")
    plt.xlabel("dim1")
    plt.ylabel("dim2")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1) 生成两组 latent
    X_sup, X_ce, y_coarse, resting_class, sub_rest = build_synthetic_latent()

    # 2) 画 latent 散点图（只看 resting 子集）
    plot_latent_resting(X_sup, X_ce, y_coarse, resting_class, sub_rest)

    # 3) 组织成 X_all，跑 panel A
    X_all = {
        "SupCon": X_sup,
        "CEonly": X_ce,
    }

    res = panel_A_metrics(
        X_all=X_all,
        y_coarse=y_coarse,
        resting_class=resting_class,
        sub_rest=sub_rest,
        out_png="panelA_demo.png",
        out_csv="panelA_demo.csv",
    )

    print("=== CE-only metrics ===")
    for k, v in res["ce"].items():
        print(f"{k:12s}: {v}")

    print("\n=== SupCon metrics ===")
    for k, v in res["sup"].items():
        print(f"{k:12s}: {v}")

    print("\nSaved figure to:", res["png"])
    print("Saved csv to   :", res["csv"])
