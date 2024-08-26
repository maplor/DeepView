
import os
import matplotlib.pyplot as plt
import numpy as np
# from skimage.transform import resize
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
# from datetime import datetime
import pandas as pd

def extract_save_all_maps(
    config,
    gputouse=None,
    Indices=None,
    modelprefix="",
    dest_folder=None,
):
    '''
    get necessary data from test sets to plot figures
    :param config:
    :param trainingsetindex:
    :param comparisonbodyparts:
    :param extract_paf:
    :param all_paf_in_one:
    :param gputouse:
    :param rescale:
    :param Indices: filename of test set
    :param modelprefix: related to model name
    :param dest_folder:
    :return:
    '''
    from deepview.utils.auxiliaryfunctions import (
        read_config,
        attempt_to_make_folder,
        get_evaluation_folder,
        # intersection_of_body_parts_and_ones_given_by_user,
    )
    from tqdm import tqdm
    from deepview.clustering_pytorch.config import load_config

    cfg = read_config(config)

    # 找到model structure，data，evaluate model
    data = extract_maps(
        config, gputouse, Indices, modelprefix
    )  # return dict

    print("Saving plots...")  # comparisonbodyparts算出数据后开始画图
    dest_path = ''
    timestamp = np.array([])
    for frac, values in data.items():  # data is a dictionary
        if not dest_folder:
            dest_folder = os.path.join(
                cfg["project_path"],
                str(get_evaluation_folder(cfg, modelprefix=modelprefix)),
                "maps",
            )  # 'evaluation-results/iteration-0/dDec7-trainset95shuffle1/maps'
        attempt_to_make_folder(dest_folder)
        filepath = "{imname}_{map}_{label}_{frac}.png"
        dest_path = os.path.join(dest_folder, filepath)

        # get whole un-overlap timestamp
        if frac == 'timestamp':
            timestamp = np.concatenate(values).reshape(-1)

    # plot
    cluster_labels = []
    for frac, values in data.items():  # data is a dictionary
        # Maps['representation'] = representation_list
        # Maps['rawdata'] = sample_list
        # Maps['timestamp'] = timestamp_list

        if frac == 'representation':
            latents_day1 = np.concatenate(values)
            reshaped_latent_day1 = latents_day1.reshape(latents_day1.shape[0], -1)  # out=len*1024
            # Reduce data dimensions using t-SNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=12)
            data_tsne = tsne.fit_transform(reshaped_latent_day1)  # 3min
            # print(data_tsne.shape)
            plt.figure()
            # ax = sns.kdeplot(data_tsne[:, 0], data_tsne[:, 1], shade=True, shade_lowest=False)
            ax = sns.kdeplot(x=data_tsne[:, 0], y=data_tsne[:, 1], zorder=0, n_levels=6, fill=True,
                        cbar=True, thresh=0.05, cmap='viridis')
            # sns.kdeplot(x=x, y=y, zorder=0, n_levels=6, shade=True,
            #             cbar=True, shade_lowest=False, cmap='viridis')
            temp = dest_path.format(
                imname='filename',
                map="scmap",
                label='density',
                frac=frac,
            )
            plt.savefig(temp)
            plt.close()

            # add cluster labels
            plt.figure()
            # cluster:
            clustering = DBSCAN(eps=2, min_samples=11).fit(data_tsne)
            # Get cluster labels for each data point
            cluster_labels = clustering.labels_
            nlabels = len(set(cluster_labels))

            for i in range(nlabels):
                plt.scatter(data_tsne[cluster_labels == i, 0], data_tsne[cluster_labels == i, 1], label=f'Cluster {i}',
                            alpha=0.7)
            plt.title('Clustered Data, DBSCAN labels')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            # plt.legend()
            temp = dest_path.format(
                imname='filename',
                map="scmap",
                label='DBSCAN',
                frac=frac,
            )
            plt.savefig(temp)
            plt.close()

        # elif frac == 'rawdata':
            raw_data_list = data['rawdata']
            raw_day = np.concatenate(raw_data_list)
            raw_day_reshape = raw_day.reshape(-1, raw_day.shape[-1])
            timestr = [pd.Timestamp(x.tolist(), unit='us') for x in timestamp]

            # remove overlap
            df = pd.DataFrame({"t": timestr})
            df_unique = df.drop_duplicates()
            removed_indices = df[~df.duplicated()]["t"].index.tolist()
            timestr_uniq = df_unique.values
            rawdata_time_uniq = raw_day_reshape[removed_indices]

            plt.figure(figsize=(10, 3))
            f, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(timestr_uniq, rawdata_time_uniq, linestyle='-')
            # cluster label on data points
            # plt.subplots(2, 1)
            # plot cluster labels
            # cluster_labels_ex = np.expand_dims(cluster_labels, axis=1)
            cluster_labels_ex = np.tile(cluster_labels[:, np.newaxis], raw_day.shape[1]).reshape(-1)
            cluster_labels_ex_uniq = list(cluster_labels_ex[removed_indices])
            # plot_label_duration(list(cluster_labels_ex_uniq), plt)
            current_label = None
            for i, label in enumerate(cluster_labels_ex_uniq + [None]):
                if label != current_label:
                    if current_label is not None:
                        ax2.fill_betweenx(y=[current_label - 0.4, current_label + 0.4],
                                          x1=start_index, x2=i, color=f"C{current_label + 1}", edgecolor="black")
                    if label is not None:
                        start_index = i
                        current_label = label
            ax2.set_yticks(range(1, max(cluster_labels_ex_uniq) + 1))
            ax2.set_xlabel("Index")
            ax2.set_ylabel("Label ID")
            ax2.set_title("Label Duration Plot")

            temp = dest_path.format(
                imname='filename',
                map="scmap",
                label='DBSCAN',
                frac='example',
            )
            plt.savefig(temp)
            plt.close()

        plt.close("all")


def plot_label_duration(labels, plt):
    current_label = None
    start_index = 0

    # Plotting
    # fig, ax = plt.subplots()
    last_token = None
    for i, label in enumerate(labels + [last_token]):
        if label != current_label:
            if current_label != last_token:
                plt.fill_betweenx(y=[current_label - 0.4, current_label + 0.4],
                                 x1=start_index, x2=i, color=f"C{current_label+1}", edgecolor="black")

            if label != last_token:
                start_index = i
                current_label = label

    plt.yticks(range(1, max(labels) + 1))
    plt.xlabel("Index")
    plt.ylabel("Label ID")
    plt.title("Label Duration Plot")

    return

def visualize_scoremaps(image, scmap):
    ny, nx = np.shape(image)[:2]
    fig, ax = form_figure(nx, ny)
    ax.plot(image)
    ax.plot(scmap, alpha=0.5)
    return fig, ax

def form_figure(nx, ny):
    fig, ax = plt.subplots(frameon=False)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.axis("off")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig, ax

def extract_maps(
    config,
    gputouse=None,
    rescale=False,
    Indices=None,
    modelprefix="",
):
    """
    Extracts the scoremap, locref, partaffinityfields (if available).

    Returns a dictionary
    read data and model structure in this function
    ----------
    config : string
        Full path of the config.yaml file as a string.

    predict: model structure

    """
    # # from deeplabcut.utils.auxfun_videos import imread, imresize
    # from deeplabcut.pose_estimation_tensorflow.core1 import (
    #     predict,
    #     predict_multianimal as predictma,
    # )
    # from deepview.clustering_pytorch.config import load_config
    # from deepview.clustering_pytorch.core1.evaluate import AE_eval_time_series

    # from deeplabcut.pose_estimation_tensorflow.datasets.utils import data_to_input
    from deepview.utils import auxiliaryfunctions
    # from deepview.clustering_pytorch.core1.evaluate import (
    #     load_model,
    # )
    # from tqdm import tqdm
    #
    # import pandas as pd
    # from pathlib import Path
    # import numpy as np
    import pickle

    start_path = os.getcwd()
    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)

    if gputouse is not None:  # gpu selectinon
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gputouse)

    # # Loading human annotatated data
    # labeleddatafolder = auxiliaryfunctions.get_labeled_data_folder(cfg)  # training-datasets/iter-0/Unaugment
    #
    # tmppath =os.path.join(
    #             cfg["project_path"],
    #             str(labeleddatafolder),
    #         )
    # with open(tmppath, 'rb') as f:
    #     RawData = pickle.load(f)
    #
    # # Make folder for evaluation
    # auxiliaryfunctions.attempt_to_make_folder(
    #     str(cfg["project_path"] + "/evaluation-results/")
    # )


    #
    # ##################################################
    # # Load and setup CNN part detector
    # ##################################################
    # datafn, metadatafn = auxiliaryfunctions.get_data_and_metadata_filenames(
    #     labeleddatafolder, cfg
    # )  # trainingsetfolder：training-datasets/iter../Unaug..
    # # datafn:training/.../d.mat; metadatafn:training/.../Docu.pickle
    # modelfolder = os.path.join(
    #     cfg["project_path"],
    #     str(
    #         auxiliaryfunctions.get_model_folder(
    #             cfg, modelprefix=modelprefix
    #         )
    #     ),
    # )  # modelfolder:Users/.../dlc-models/iter/dshuffle1(在training时新生成的)
    # path_test_config = Path(modelfolder) / "test" / "model_cfg.yaml"
    #
    # try:
    #     dlc_cfg = load_config(str(path_test_config))
    # except FileNotFoundError:
    #     raise FileNotFoundError(
    #         "It seems the model does not exist."
    #     )
    # # change batch size, if it was edited during analysis!
    # dlc_cfg["batch_size"] = 1  # in case this was edited for analysis.
    #
    # Create folder structure to store results.evaluationfolder：evaluation-results/iter/dDec..
    evaluationfolder = os.path.join(
        cfg["project_path"],
        str(
            auxiliaryfunctions.get_evaluation_folder(
                cfg, modelprefix=modelprefix
            )
        ),
    )  # '/Users/cassie/Desktop/d-v-2023-12-04/evaluation-results/iteration-0/dDec4'
    auxiliaryfunctions.attempt_to_make_folder(evaluationfolder, recursive=True)
    # path_train_config = modelfolder / 'train' / 'pose_cfg.yaml'

    # get data
    Maps = {}  # 最后返回这个dictionary
    rawdata_files = auxiliaryfunctions.get_labeled_data_folder(cfg)
    # todo 目前只选择一天的数据
    with open(os.path.join(rawdata_files[0].parent, rawdata_files[0].stem+'_toplot.pkl'), 'rb') as f:
        [representation_list, sample_list, timestamp_list, label_list, domain_list] = pickle.load(f)

    Maps['representation'] = representation_list
    Maps['rawdata'] = sample_list
    Maps['timestamp'] = timestamp_list

    os.chdir(str(start_path))  # Change the current working directory to the specified path
    return Maps  # 看运行到哪。。。。
