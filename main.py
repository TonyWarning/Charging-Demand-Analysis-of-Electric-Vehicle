import numpy as np
import pandas as pd
from pandas import read_csv
import os
from chardet.universaldetector import UniversalDetector
import time
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer


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

def train_test_split(fileadd):
    def xySplit(fileadd):
        data = read_csv(fileadd, encoding='utf-8', error_bad_lines=False)

        # delete null data
        data = data.dropna(axis=0, how='any')

        # 将数据分为x和y
        x = data.drop(['category'], axis=1)
        y = data['category']

        # save data
        x.to_csv(diradd + '/x.csv', encoding='utf-8', index=False)
        y.to_csv(diradd + '/y.csv', encoding='utf-8', index=False)

    def dataSplit(xAdd, yAdd):
        x = read_csv(xAdd, encoding='utf-8', error_bad_lines=False)
        y = read_csv(yAdd, encoding='utf-8', error_bad_lines=False)

        # split data
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        # save data
        x_train.to_csv(diradd + '/x_train.csv', encoding='utf-8', index=False)
        x_test.to_csv(diradd + '/x_test.csv', encoding='utf-8', index=False)
        y_train.to_csv(diradd + '/y_train.csv', encoding='utf-8', index=False)
        y_test.to_csv(diradd + '/y_test.csv', encoding='utf-8', index=False)

    xySplit(fileadd)
    dataSplit(diradd + '/x.csv', diradd + '/y.csv')

def train_testModel(x_trainAdd, y_trainAdd, x_testAdd, y_testAdd):
    x_train = read_csv(x_trainAdd, encoding='utf-8', error_bad_lines=False)
    y_train = read_csv(y_trainAdd, encoding='utf-8', error_bad_lines=False)
    x_test = read_csv(x_testAdd, encoding='utf-8', error_bad_lines=False)
    y_test = read_csv(y_testAdd, encoding='utf-8', error_bad_lines=False)

    def normalization(x):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)
        return x

    # train model
    def training_testing(x_train, y_train, x_test, y_test):

        from sklearn.metrics import classification_report
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score

        # # training
        # from sklearn.ensemble import RandomForestClassifier
        # from sklearn.metrics import accuracy_score
        # clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
        # clf.fit(x_train, y_train)
        #
        # # save model
        # import joblib
        # joblib.dump(clf, diradd + '/model.pkl')

        ### 随机森林
        print("==========================================")
        RF = RandomForestClassifier(n_estimators=10, random_state=11)
        RF.fit(x_train, y_train)
        predictions = RF.predict(x_test)
        print("RF")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### Logistic Regression Classifier
        print("==========================================")
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(penalty='l2')
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("LR")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### Decision Tree Classifier
        print("==========================================")
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("DT")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### GBDT(Gradient Boosting Decision Tree) Classifier
        print("==========================================")
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=200)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("GBDT")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ###AdaBoost Classifier
        print("==========================================")
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier()
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("AdaBoost")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### GaussianNB
        print("==========================================")
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("GaussianNB")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### Linear Discriminant Analysis
        print("==========================================")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf = LinearDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("Linear Discriminant Analysis")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### Quadratic Discriminant Analysis
        print("==========================================")
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        clf = QuadraticDiscriminantAnalysis()
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("Quadratic Discriminant Analysis")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### SVM Classifier
        print("==========================================")
        from sklearn.svm import SVC
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("SVM")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### Multinomial Naive Bayes Classifier
        print("==========================================")
        from sklearn.naive_bayes import MultinomialNB
        clf = MultinomialNB(alpha=0.01)
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("Multinomial Naive Bayes")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### xgboost
        import xgboost
        print("==========================================")
        from sklearn.naive_bayes import MultinomialNB
        clf = xgboost.XGBClassifier()
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("xgboost")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

        ### voting_classify
        from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier
        import xgboost
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        clf1 = GradientBoostingClassifier(n_estimators=200)
        clf2 = RandomForestClassifier(random_state=0, n_estimators=500)
        # clf3 = LogisticRegression(random_state=1)
        # clf4 = GaussianNB()
        clf5 = xgboost.XGBClassifier()
        clf = VotingClassifier(estimators=[
            # ('gbdt',clf1),
            ('rf', clf2),
            # ('lr',clf3),
            # ('nb',clf4),
            # ('xgboost',clf5),
        ],
            voting='soft')
        clf.fit(x_train, y_train)
        predictions = clf.predict(x_test)
        print("voting_classify")
        print(classification_report(y_test, predictions))
        print("AC", accuracy_score(y_test, predictions))

    x_train = normalization(x_train)
    x_test = normalization(x_test)
    print("start training")
    training_testing(x_train, y_train, x_test, y_test)
    print("finish training")

if __name__ == '__main__':
    # Change to UTF-8 and concat all csv files
    diradd = 'D:\Data-Charging-Demand-Analysis-of-EV'
    # preWork(diradd)

    # # Data preprocessing
    # fileadd = 'D:/Data-Charging-Demand-Analysis-of-EV/finalData.csv'
    # dataPreprocessing(fileadd)

    # # train_test_split
    # fileadd = 'D:/Data-Charging-Demand-Analysis-of-EV/preprocessed_data.csv'
    # train_test_split(fileadd)

    # train model
    x_trainAdd = 'D:/Data-Charging-Demand-Analysis-of-EV/x_train.csv'
    y_trainAdd = 'D:/Data-Charging-Demand-Analysis-of-EV/y_train.csv'
    x_testAdd = 'D:/Data-Charging-Demand-Analysis-of-EV/x_test.csv'
    y_testAdd = 'D:/Data-Charging-Demand-Analysis-of-EV/y_test.csv'
    train_testModel(x_trainAdd, y_trainAdd, x_testAdd, y_testAdd)






