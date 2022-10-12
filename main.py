import numpy as np
import pandas as pd
from pandas import read_csv
import os
from chardet.universaldetector import UniversalDetector

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


if __name__ == '__main__':
    # Change to UTF-8 and concat all csv files
    # diradd = 'D:\Data-Charging-Demand-Analysis-of-EV'
    # preWork(diradd)

    fileadd = 'D:/Data-Charging-Demand-Analysis-of-EV/topic5_2108_1.csv'



