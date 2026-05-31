import os
import pickle
from pathlib import Path

import pandas as pd

from deepview.utils import conversioncode


def _robust_path_split(path):
    sep = "\\" if "\\" in path else "/"
    splits = path.rsplit(sep, 1)
    if len(splits) == 1:
        parent = "."
        file = splits[0]
    elif len(splits) == 2:
        parent, file = splits
    else:
        raise ("Unknown filepath split for path {}".format(path))
    filename, ext = os.path.splitext(file)
    return parent, filename, ext


def merge_annotateddatasets(cfg, trainingsetfolder_full):
    """
    Merges all the h5 files for all labeled-datasets (from individual videos).

    This is a bit of a mess because of cross platform compatibility.

    Within platform comp. is straightforward. But if someone labels on windows and wants to train on a unix cluster or colab...

    #------by xia---------
    the csv file contains every sample filename and the label positions (by human)
    """

    AnnotationData = []
    data_path = Path(os.path.join(cfg["project_path"], "labeled-data"))
    files = cfg["file_sets"].keys()

    ## removed for loop, assume only one file
    _, filename, _ = _robust_path_split(files)
    file_path = os.path.join(
        data_path / filename, f'CollectedData_{cfg["scorer"]}.pkl'
    )
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        conversioncode.guarantee_multiindex_rows(data)
        if data.columns.levels[0][0] != cfg["scorer"]:
            print(
                f"{file_path} labeled by a different scorer. This data will not be utilized in training dataset creation. If you need to merge datasets across scorers, see https://github.com/DeepLabCut/DeepLabCut/wiki/Using-labeled-data-in-DeepLabCut-that-was-annotated-elsewhere-(or-merge-across-labelers)"
            )
        AnnotationData.append(data)
    except FileNotFoundError:
        print(file_path, " not found (perhaps not annotated).")

    if not len(AnnotationData):
        print(
            "Annotation data was not found by splitting video paths (from config['video_sets']). An alternative route is taken..."
        )
        AnnotationData = conversioncode.merge_windowsannotationdataONlinuxsystem(cfg)
        if not len(AnnotationData):
            print("No data was found!")
            return

    AnnotationData = pd.concat(AnnotationData).sort_index()

    # When concatenating DataFrames with misaligned column labels,
    # all sorts of reordering may happen (mainly depending on 'sort' and 'join')
    # Ensure the 'bodyparts' level agrees with the order in the config file.
    bodyparts = cfg["bodyparts"]
    AnnotationData = AnnotationData.reindex(
        bodyparts, axis=1, level=AnnotationData.columns.names.index("bodyparts")
    )

    # save data, data is dataframe, include raw data and annotations (by human)
    # (deeplabcut) row: sample name, column: label types
    # (deepview) todo: change the dataframe, I want to label the start and end time of activities for each channel
    filename = os.path.join(trainingsetfolder_full, f'CollectedData_{cfg["scorer"]}')
    # AnnotationData.to_hdf(filename + ".h5", key="df_with_missing", mode="w")
    with open(filename + ".pkl", 'wb') as f:
        pickle.dump(AnnotationData, f)
    # human readable. human labels of every samples
    AnnotationData.to_csv(filename + ".csv")

    return AnnotationData
    # return splits