import os
import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


logger = logging.getLogger(__name__)


def preprocess_sensor_data(df,
                           clipping=True,
                           clipping_threshold=8,
                           method="none",
                           check_df=False):
    '''
    clipping using clip method (pandas)
    standardization using sklearn.preprocessing.StandardScaler()
    interpolation using scipy.interpolate.interp1d()
    '''

    if len(df) == 0:
        logger.warning("No sensor data")
    else:
        # clipping (just to make sure that measurement errors are removed)
        # Note: ±8G
        # # todo: 真的需要吗？阈值怎么确定？
        # if clipping == True:
        #     df["acc_x"] = df["acc_x"].clip(lower=-clipping_threshold,
        #                                    upper=clipping_threshold)
        #     df["acc_y"] = df["acc_y"].clip(lower=-clipping_threshold,
        #                                    upper=clipping_threshold)
        #     df["acc_z"] = df["acc_z"].clip(lower=-clipping_threshold,
        #                                    upper=clipping_threshold)

        # if check_df == True:
        #     display(df[:5])

        # Note: we implemented the below pre-processing methods,
        # but did not use the standardization nor normalizetion in dl-wabc study
        if method == "standardization":
            scaling_columns = ["acc_x", "acc_y", "acc_z"]
            sensor_data = df[scaling_columns]
            scaler = StandardScaler().fit(sensor_data.values)
            scaled_sensor_data = scaler.transform(sensor_data.values)
            df[scaling_columns] = scaled_sensor_data
            # if check_df == True:
            #     display(df[:5])
        elif method == "normalization":
            scaling_columns = ["acc_x", "acc_y", "acc_z"]
            sensor_data = df[scaling_columns]
            scaler = MinMaxScaler().fit(sensor_data.values)
            scaled_sensor_data = scaler.transform(sensor_data.values)
            df[scaling_columns] = scaled_sensor_data
            # if check_df == True:
            #     display(df[:5])

    return df


def save_preprocessed_data(df, output_dir_path, species, animal_id, label_id_path):
    '''
    Args:
        df (DataFrame): a preprocess data frame
        output_dir_path (str): a path to save the preprocessed data frame
        species (str): species name, such as omizunagidori, umineko
                        (Japanese names of streaked shearwaters and black-tailled gulls)
        animal_id (str): animal id, such as OM2101, UM1901
        label_id_path: label file path

    Return:
        None
    '''

    if len(df) == 0:
        logger.warning("No sensor data; no data saved")
    else:
        df = df.loc[:, ['datetime', 'unixtime',
                        'acc_x', 'acc_y', 'acc_z', 'label']]
        # if species == "omizunagidori":
        #     label_id_path = "../data/id_files/label_id_omizunagidori.csv"
        # elif species == "umineko":
        #     label_id_path = "../data/id_files/label_id_umineko.csv"

        df["label"] = df["label"].astype("object")
        df_label_id = pd.read_csv(label_id_path, low_memory=False)
        df_label_id["label"] = df_label_id["label"].astype("object")
        df_merge = pd.merge(df, df_label_id[["label", "label_id"]],
                            how='left', on='label')
        df = df_merge

        save_dir = os.path.join(output_dir_path, species)
        if os.path.exists(save_dir) == False:
            os.makedirs(save_dir)

        df_save_path = os.path.join(save_dir, str(animal_id) + ".csv")

        df.to_csv(df_save_path, index=False)
        logger.info("Preprocessed data saved: %s", df_save_path)
