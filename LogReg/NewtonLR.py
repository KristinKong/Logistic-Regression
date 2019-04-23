'''牛顿法实现逻辑回归'''
# -*- coding: utf-8 -*-
import numpy as np
import random as rd
import pickle as pk
import matplotlib.pyplot as plt
from codecs import open as cd_open
import sys

# 定义绘图所需的对象和操作
class DrawFeature:

    # 初始化函数
    def __init__(self):
        self.feature = 0
        self.pos_name = 0      # 正例名称
        self.neg_name = 0      # 反例名称
        self.feature_name = [] # 每个特征的名字
        self.feature_min = []  # 每个特征的最小值
        self.feature_max = []  # 每个特征的最大值
        self.x_index = 0  # 选择绘图的x轴特征下标
        self.y_index = 0  # 选择绘图的y轴特征下标

    # 从配置文件中读取并初始化绘图参数
    def getDrawParameter(self, feat, name, min, max):
        self.feature = feat
        for i in range(0, self.feature):
            self.feature_name.append(name[i])
            self.feature_max.append(float(max[i]))
            self.feature_min.append(float(min[i]))

    # 将正反列下标区分开并获得绘制的散点x,y坐标
    def getDrawIndex(self, X, Y, sample_num, subplot, beta, title):
        pos_x_cord = []
        pos_y_cord = []
        neg_x_cord = []
        neg_y_cord = []
        for i in range(0, sample_num):  # 分类散点
            if Y[i] == 1.0:  # 正例散点
                pos_x_cord.append(X[i, self.x_index])
                pos_y_cord.append(X[i, self.y_index])
            else:
                neg_x_cord.append(X[i, self.x_index])
                neg_y_cord.append(X[i, self.y_index])
        p1 = subplot.scatter(pos_x_cord, pos_y_cord, s=10, c="red", marker='s')
        p2 = subplot.scatter(neg_x_cord, neg_y_cord, s=10, c="blue", marker='x')
        step = float(self.feature_max[self.x_index]-1.0)/8.0
        x = np.arange(self.feature_min[self.x_index]-0.3, self.feature_max[self.x_index]+0.7, step)  # 绘制直线,生成起始下标和步长
        b = beta[-1]-0.5            # 常数项决策边界概率为0.5
        k = beta[self.x_index]      # 斜率
        c = beta[self.y_index]      # y值对应的beta
        y = (k*x+b)/(-c)
        y = y.getA1()
        subplot.set_xlim(self.feature_min[self.x_index]-0.4,self.feature_max[self.x_index]+1)
        subplot.set_ylim(self.feature_min[self.y_index]-0.4,self.feature_max[self.y_index]+1)
        subplot.legend(handles = [p1, p2], labels = [self.pos_name, self.neg_name], loc = 'upper right')
        subplot.set_title(title)
        subplot.set_xlabel(self.feature_name[self.x_index])
        subplot.set_ylabel(self.feature_name[self.y_index])
        subplot.plot(x, y)

    # 根据配置文件中抽取的特征下标x_index和y_index绘制出有关X/Y的散点图
    def drawFeatures(self, TrainX, TrainY, TestX, TestY, train_sample, test_sample, beta):
        plt.figure(num=0, figsize=(10, 5))
        figure = plt.figure(0)      # 获得当前画布
        train_plot = figure.add_subplot(1, 2, 1)  # (i,j,k)将图像分为(i*j)块，该图像在第k块
        test_plot = figure.add_subplot(1, 2, 2)
        self.getDrawIndex(TrainX, TrainY, train_sample, train_plot, beta, "Train Set")
        self.getDrawIndex(TestX, TestY, test_sample, test_plot, beta, "Test Set")
        plt.show()


# 定义逻辑回归类函数
class LogReg:

    def __init__(self):
        self.train_sample = 0  # 训练集样本容量
        self.test_sample = 0  # 测试集样本容量
        self.feature = 0  # 数据特征向量维度
        self.type_sum = 0  # 数据集中所含样本种类
        self.type_num = []  # 每个类别所含样本数目列表
        self.type_start_index = []  # 每个类别起始下标列表
        self.type_name = []  # 每个类别的名称
        self.pos_index = 0  # 正例起始下标
        self.neg_index = 0  # 反例起始下标
        self.file_name = 0  # 获取读入文件的路径
        self.threshold = 0  # 迭代损失函数阈值
        self.iteration = 0  # 迭代的最大次数
        self.draw = DrawFeature()  # 绘图函数
        self.beta = None    # 预测矩阵
        self.TrainX = None  # 训练集输入矩阵
        self.TrainY = None  # 训练集标签
        self.TestX = None   # 测试集输入矩阵
        self.TestY = None   # 测试集标签

    # 读配置文件获取代码运行的相关参数
    def readConfig(self):
        f = cd_open("config.cfg",'r', encoding="utf-8")
        conf = f.readlines()
        temp = conf[0].split(',')     # 获取数据集
        self.file_name = temp[0]
        temp = conf[1].split(',')     # 获取数据集所含类别总数，特征向量维度，训练集/测试集样本容量
        self.type_sum = int(temp[0])
        self.feature = int(temp[1])
        self.train_sample = int(temp[2])
        self.test_sample = int(temp[3])
        temp = conf[2].split(',')       # 获取每个类别的名称
        temp1 = conf[3].split(',')     # 获取每个类别所含的样本容量
        temp2 = conf[4].split(',')      # 获取每个类别样本容量的数目/起始下标
        for i in range(0, self.type_sum):
            self.type_name.append(temp[i])
            self.type_num.append(int(temp1[i]))
            self.type_start_index.append(int(temp2[i]))
        temp = conf[5].split(',')     # 获取每个类别样本容量的数目/起始下标
        self.pos_index = int(temp[0])
        self.neg_index = int(temp[1])
        self.draw.x_index = int(temp[2])
        self.draw.y_index = int(temp[3])
        temp = conf[6].split(',')     # 获取损失函数阈值和最大迭代次数
        self.threshold = float(temp[0])
        self.iteration = int(temp[1])
        temp = conf[7].split(',')      # 获取每类特征的名称,用于作图
        temp1 = conf[8].split(',')     # 获取每类特征的最小值，用于作图
        temp2 = conf[9].split(',')     # 获取每类特征的最大值，用于作图
        self.draw.getDrawParameter(self.feature, temp, temp1, temp2)
        self.draw.pos_name = self.type_name[self.pos_index]
        self.draw.neg_name = self.type_name[self.neg_index]
        f.close()

    # 获得随机抽取数据的下标 （tp->类别编号 samples->抽取实例个数 max_lable->每个类别最大下标）
    def getIndex(self, tr_start, tr_end,te_start, te_end, type_index, all_sample, label):
        index_list = []
        for i in range(tr_start, tr_end):  # 生成训练集
            index = rd.randint(0, self.type_num[type_index])
            while index in index_list:
                index = rd.randint(0, self.type_num[index])
            temp = all_sample[index + self.type_start_index[type_index]].split(',')
            temp[-1] = 1  # 最后一个分类填入beta0构成矩阵运算
            self.TrainX[i, :] = temp
            self.TrainY[i, 0] = label
        for i in range(te_start, te_end):  # 生成测试集
            index = rd.randint(0, self.type_num[type_index])
            while index in index_list:
                index = rd.randint(0, self.type_num[index])
            temp = all_sample[index + self.type_start_index[type_index]].split(',')
            temp[-1] = 1  # 最后一个分类填入beta0构成矩阵运算
            self.TestX[i, :] = temp
            self.TestY[i, 0] = label

    # 写文件操作，将预处理之后的数据存为pkl文件，以后使用就可以直接读取
    def writePkl(self, filename, lst):
        fout = open(filename, 'wb')    # pkl文件要采用流写入方式
        pk.dump(lst, fout)
        fout.close()

    # 读文件操作，将可以直接使用的pkl文件读入内存
    def readPkl(self, filename):
        fin = open(filename, 'rb')
        lst = pk.load(fin)
        fin.close()
        return lst

    # 数据初始化处理，从给定配置文件中读取数据集中提取训练集和测试集(samples->样本容量 freatures->特征数量 type1->抽取类别1 type2->抽取类别2)
    def first_preProcess(self):
        self.readConfig()
        f = open(self.file_name, 'r')  # 将样例读入缓存
        all_sample = f.readlines()
        f.close()
        self.feature += 1
        self.beta = np.mat(np.zeros((self.feature, 1)))  # beta,beta0构成的列向量,赋0-1随机初始值,二维数组要用二维括号输出
        self.TrainX = np.mat(np.zeros((self.train_sample, self.feature)))
        self.TrainY = np.mat(np.zeros((self.train_sample, 1)))
        self.TestX = np.mat(np.zeros((self.test_sample, self.feature)))
        self.TestY = np.mat(np.zeros((self.test_sample, 1)))
        tr = int(self.train_sample / 2)
        te = int(self.test_sample / 2)
        self.getIndex(0, tr, 0, te, self.pos_index, all_sample, 1)         # 在随机取训练集时保证等比例抽取（正）
        tr = int((self.train_sample+1) / 2)
        te = int((self.test_sample+1) / 2)
        self.getIndex(tr, self.train_sample, te, self.test_sample,self.neg_index, all_sample, 0)  # 在随机取训练集时保证等比例抽取（反）
        # 调用预处理之后产生可以直接进行运算的矩阵，可以存为pkl文件下次就直接读入
        self.writePkl("TranX.pkl", self.TrainX)
        self.writePkl("TranY.pkl", self.TrainY)
        self.writePkl("TestX.pkl", self.TestX)
        self.writePkl("TestY.pkl", self.TestY)

    # 数据初始化处理，从生成的pkl文件中读取
    def second_preProcess(self):
        self.readConfig()
        self.feature += 1
        self.beta = np.mat(np.zeros((self.feature, 1)))  # beta,beta0构成的列向量,赋0-1随机初始值
        self.TrainX = self.readPkl("TranX.pkl")
        self.TrainY = self.readPkl("TranY.pkl")
        self.TestX = self.readPkl("TestX.pkl")
        self.TestY = self.readPkl("TestY.pkl")

    # 计算输入特征向量的回归函数 (beta->预测矩阵  X->输入向量)
    def logistic(self, X):
        return 1.0 / (1 + np.exp(-X * self.beta))

    # 计算海塞矩阵，梯度以及损失函数 (beta->预测矩阵  X->输入向量 Y->类别向量)
    def getComputeData(self):
        P = self.logistic(self.TrainX)  # 计算预测类别向量P
        G = 1.0/ self.train_sample*self.TrainX.T * (P - self.TrainY)  # 计算梯度
        H = 1.0/ self.train_sample*self.TrainX.T * np.diag(P.getA1() * (1 - P).getA1()) * self.TrainX  # 计算海塞矩阵
        loss = np.sum(-self.TrainY.getA1() * np.log(P.getA1()) - (1.0 - self.TrainY).getA1() * np.log((1 - P).getA1()))  # 计算损失(取反为正数方便处理)
        return G, H, loss

    # 用牛顿法实现逻辑回归   (X->输入向量 Y->类别向量 threshold->损失阈值 iterations->迭最大代次数)
    def newtonLogistic(self):
        loss0 = float('inf')
        for i in range(0, self.iteration):
            try:
                G, H, loss = self.getComputeData()
                self.beta = self.beta - H.I * G
                i += 1
                if loss0 - loss < self.threshold:  # 如果损失函数提前达到阈值，则结束
                    break
                loss0 = loss
            except Exception as e:  # 若海塞矩阵奇异，需要加一个很小的常数进行处理
                H += 0.0001
                break

    # 利用测试集进行预测并计算准确率
    def predictResult(self):
        sum = 0
        TempY = self.logistic(self.TestX)
        for i in range(0, len(self.TestY)):
            res = abs(TempY[i]-self.TestY[i])
            if res < 0.5: # 判断正确的情况
                sum += 1
            elif res is 0.5:
                jud = rd.randint(0,1)      # 生成0/1
                if jud == self.TestY[i]:
                    sum += 1
        precison = float(sum)/float(self.test_sample)
        print("预测准确率为 = ", precison)

    # 执行函数
    def excuteLR(self, jud):
        if jud =="RAW":
            print("开始处理数据并划分测试集和训练集")
            self.first_preProcess()
        else:
            print("直接利用处理过的数据")
            self.second_preProcess()
        print("开始迭代回归")
        self.newtonLogistic()
        print("开始进行预测")
        self.predictResult()
        self.draw.drawFeatures(self.TrainX, self.TrainY, self.TestX, self.TestY, \
                               self.train_sample, self.test_sample, self.beta)


if __name__ == "__main__":
    print("开始进行牛顿法实现逻辑回归" )
    LR = LogReg()
    LR.excuteLR("RAW")  # RAW表示需要处理数据集，若为其它则直接读取pkl文件















