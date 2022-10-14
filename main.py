import numpy as np
import pandas as pd
from pandas import read_csv
import os
from chardet.universaldetector import UniversalDetector
import time
import sklearn

def preWork(diradd):

    def changeToUtf8(diradd):
        fileSuffix = '.csv'
        fns = []
        for dirpath, dirnames, filenames in os.walk(diradd):
            for filename in filenames:
                if filename.endswith(fileSuffix):
                    fns.append(os.path.join(dirpath, filename))

        def read_file(file):
            with open(file, 'rb') as f:
                return f.read()

        def get_encode_info(file):
            """
            逐个读取文件的编码方式
            """
            with open(file, 'rb') as f:
                detector = UniversalDetector()
                for line in f.readlines():
                    detector.feed(line)
                    if detector.done:
                        break
                detector.close()
                return detector.result['encoding']

        def convert_encode2utf8(file, original_encode, des_encode):
            """
            将文件的编码方式转换为utf-8，并写入原先的文件中。
            """
            file_content = read_file(file)
            file_decode = file_content.decode(original_encode, 'ignore')
            file_encode = file_decode.encode(des_encode)
            with open(file, 'wb') as f:
                f.write(file_encode)

        fileNum = 0
        for filename in fns:
            try:
                file_content = read_file(filename)
                encode_info = get_encode_info(filename)
                if encode_info != 'utf-8':
                    fileNum += 1
                    convert_encode2utf8(filename, encode_info, 'utf-8')
                    print('成功转换 %s 个文件 %s ' % (fileNum, filename))
            except BaseException:
                print(filename, '存在问题，请检查！')

    # 转换为utf8编码
    #   changeToUtf8(diradd)

    # 导入数据
    finalData = pd.DataFrame()
    for dirpath, dirnames, filenames in os.walk(diradd):
        for filename in filenames:
            if filename.endswith('.csv'):
                os.path.join(dirpath, filename)
                file = os.path.join(dirpath, filename)
                data = read_csv(file, encoding='utf-8', error_bad_lines=False)
                print('start concat: ', filename)
                finalData = pd.concat([finalData, data], axis=0)
                print('finish concat: ', filename)

    print(finalData)
    # 保存数据
    finalData.to_csv(os.path.join(diradd, 'finalData.csv'), encoding='utf-8', index=False)
    print('数据保存成功')

def dataPreprocessing(fileadd):
    data = read_csv(fileadd, encoding='utf-8', error_bad_lines=False)

    # 取出有用的字段
    select_cols = ['rating_energy', 'drive_range', 'start_time', 'start_soc', 'is_holiday', 'category']
    data = data[select_cols]

    # 根据category字段，10:行驶 20:停车 30:充电 筛选出 20和30的记录
    data = data[(data['category'] == 20) | (data['category'] == 30)]

    # 将category中的20和30替换为0和1
    data['category'] = data['category'].replace(20, 0)
    data['category'] = data['category'].replace(30, 1)

    # 将start_time字段转换为时间格式
    def timeTrans(timestamp):
        return (timestamp % 86400000) / 86400000
    data['start_time'] = data['start_time'].apply(timeTrans)

    # save data
    data.to_csv(diradd + '/preprocessed_data.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    # Change to UTF-8 and concat all csv files
    diradd = 'D:\Data-Charging-Demand-Analysis-of-EV'
    # preWork(diradd)

    fileadd = 'D:/Data-Charging-Demand-Analysis-of-EV/finalData.csv'
    dataPreprocessing(fileadd)




