def combine_rectangles(rectangles, threshold_seconds=100):
    if not rectangles:
        return []

    # 将矩形按开始时间排序
    rectangles.sort(key=lambda x: x[0])

    combined_rectangles = []
    current_start, current_end = rectangles[0]

    for start, end in rectangles[1:]:
        # 如果当前时间段与下一个时间段间隔小于阈值
        if (start - current_end) <= threshold_seconds:
            # 合并时间段
            current_end = max(current_end, end)
        else:
            # 否则，将当前时间段加入结果列表，并更新当前时间段
            combined_rectangles.append((current_start, current_end))
            current_start, current_end = start, end

    # 添加最后一个时间段
    combined_rectangles.append((current_start, current_end))

    return combined_rectangles


def find_charts_data_columns(sensor_dict, column_names):
    # new_column_names = []
    metadatas = []
    for column_name in column_names:
        # real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        # new_column_names.extend(real_names) # 将实际列名添加到新的列名列表中
        if column_name.upper() == "GPS":
            real_names = ['GPS_velocity', 'GPS_bearing']
        else:
            real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        # 创建元数据信息
        metadata = {
            "name": column_name,
            "xAxisName": "timestamp",
            "yAxisName": "Y Axis 1",
            "series": real_names
        }
        metadatas.append(metadata)
    return metadatas