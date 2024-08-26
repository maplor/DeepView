

import os
DEBUG = True and "DEBUG" in os.environ and os.environ["DEBUG"]
# from deepview.version import __version__, VERSION

from deepview.utils import (
    auxiliaryfunctions,
)

from deepview.create_project import (
    create_new_project,
)

from deepview.generate_training_dataset import (
    create_training_dataset,
)

# Train, evaluate & predict functions / all require TF
from deepview.clustering_pytorch import (
    train_network,
    # return_train_network_path,
    evaluate_network,
    visualizemaps,
    # return_evaluate_network_data,
    # analyze_videos,
    # create_tracking_dataset,
    # analyze_time_lapse_frames,
    # convert_detections2tracklets,
    # extract_maps,
    # visualize_scoremaps,
    # visualize_locrefs,
    # visualize_paf,
    extract_save_all_maps,
    # export_model,
    # video_inference_superanimal,
)

from deepview.supv_learning_pytorch import (
    train_sup_network,
)


# def extract_save_all_maps(config, shuffle, Indices):
#     return None