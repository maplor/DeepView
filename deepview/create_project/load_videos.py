import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

# Load the CSV file to inspect its contents
# file_path = "/Users/cassie/Downloads/LB11.csv"
file_path = r"C:\Users\dell\Desktop\a-aa-2024-10-22\videos\LB10.csv"
# file_path = r"C:\Users\dell\Desktop\xia-video-2024-10-21\videos\LB12.csv"
# file_path = r"C:\Users\user\Documents\WeChat Files\wxid_mi05poeuk7a022\FileStorage\File\2024-09\xia-san-video-sample\umineko\LB12.csv"

sqlite3_path = r"C:\Users\dell\Desktop\a-aa-2024-10-22\db\database.db"
conn = sqlite3.connect(sqlite3_path)
# conn = sqlite3.connect('database.db')
cursor = conn.cursor()

def insert_camera_info(CSV_name, cameraID, frame_rate, frame_count, start_time, end_time):
    # Convert start_time and end_time to datetime objects
    start_time_dt = datetime.strptime(start_time, "%Y%m%dT%H:%M:%S")
    end_time_dt = datetime.strptime(end_time, "%Y%m%dT%H:%M:%S")
    
    cursor.execute('''INSERT INTO videos (animal_tag, video_id, framerate, frame_count, video_stt, video_stp) VALUES (?, ?, ?, ?, ?, ?)''',
                     (CSV_name, cameraID, frame_rate, frame_count, start_time_dt, end_time_dt))
    conn.commit()


# Define the function to load a CSV file and convert it into a dataframe
def load_csv_to_dataframe(file_path):
    df = pd.read_csv(file_path)
    return df


# # Define the function to remove rows with null values and plot the CameraCount column
# def clean_and_plot_camera_count(df):
#     # Drop rows where the CameraCount column has NaN values
#     df_clean = df.dropna(subset=['CameraCount'])
#
#     # Plot the CameraCount column
#     plt.figure(figsize=(10, 6))
#     plt.plot(df_clean['CameraCount'], label='CameraCount')
#     plt.xlabel('Index')
#     plt.ylabel('CameraCount')
#     plt.title('CameraCount over Time')
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# Load the CSV file into a dataframe
df = load_csv_to_dataframe(file_path)
CSV_name = file_path.split('\\')[-1]


# # Clean and plot the CameraCount column
# clean_and_plot_camera_count(df)

# Function to extract dataframe segments for each unique value
def extract_segments_for_values(df, values):
    segments = {}
    for value in values:
        # Find the index range for the first and last occurrence of the value
        first_index = df[df['CameraCount'] == value].index.min()
        last_index = df[df['CameraCount'] == value].index.max()

        # Extract the segment from the first to the last occurrence of this value
        segment = df.loc[first_index:last_index]

        # Store the segment in a dictionary
        segments[value] = segment
    return segments

# get timestamps of every camera from csv file
camera_count_unique_values_df = df['CameraCount'].unique()
camera_count_unique_values = list(range(len(camera_count_unique_values_df) - 2))

# Extract the segments for values larger than -1
camera_count_segments = extract_segments_for_values(df, values=camera_count_unique_values)
print(camera_count_segments)

# Function to combine values from 'Month', 'Day', 'Hour', 'Min', 'Sec' columns in the given format
def combine_time_columns(row):
    month = f"{int(row['Month']):02}"
    day = f"{int(row['Day']):02}"
    hour = f"{int(row['Hour']):02}"
    minute = f"{int(row['Min']):02}"
    second = f"{int(row['Sec']):02}"
    year = "2018"  # Assuming the year is 2024, you can modify this as needed
    return f"{year}{month}{day}T{hour}:{minute}:{second}"

# extract cameraID, frame_rate, frame_count, start_time, end_time
def extract_seg_information(idx, cameradf):
    '''
    :param idx, cameradf: 一段视频对应的所有csv信息
    :return cameraID: 根据dict的key生成，对应csv的CameraCount列
    :return frame_rate: 假设frame rate固定
    :return frame_count: 假设视频从第一次出现cameraID到最后一次出现是稳定帧率，计算总帧数
    :return start_time: 由于目前时间很奇怪，需要手动定义才不会有bug，目前手动拼接成字符串
    :return end_time: 由于目前时间很奇怪，需要手动定义才不会有bug，目前手动拼接成字符串
    '''
    cameraID = idx
    frame_rate = 15
    # cameradf = camera_info_dict.values()
    frame_count = len(cameradf)
    first_row = cameradf.iloc[0]
    last_row = cameradf.iloc[-1]
    start_time = combine_time_columns(first_row)
    end_time = combine_time_columns(last_row)
    return cameraID, frame_rate, frame_count, start_time, end_time

for idx, cameradf in camera_count_segments.items():
    cameraID, frame_rate, frame_count, start_time, end_time = \
        extract_seg_information(idx, cameradf)
    print('sucessfully get camera info')
    print(cameraID, frame_rate, frame_count, start_time, end_time)
    # 插入数据库
    insert_camera_info(CSV_name, cameraID, frame_rate, frame_count, start_time, end_time)
    
    # 打印开始减去结束时间
    print(pd.to_datetime(end_time) - pd.to_datetime(start_time))


conn.close()