import os
import cv2
import glob
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import Normalize
from paddle import nn
import tkinter as tk
from tkinter import filedialog
import os


#数据读取
#推理类
class MyDataset1(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self,transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super().__init__()
        self.data_list = []
        # 获取当前脚本所在的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 创建主窗口
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filetypes = [("Image files", "*.jpg;*.jpeg;*.png;")]
        # 打开文件对话框，允许选择多个文件
        self.data_list = filedialog.askopenfilenames(filetypes=filetypes, initialdir=current_dir)
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image_path = self.data_list[index]
        # 读取灰度图
        image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        image=cv2.resize(image,(100,100))
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = image.astype('float32')
        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = 0
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)
#训练类
class MyDataset(Dataset):
    """
    步骤一：继承 paddle.io.Dataset 类
    """
    def __init__(self,label_path, transform=None):
        """
        步骤二：实现 __init__ 函数，初始化数据集，将样本和标签映射到列表中
        """
        super().__init__()
        self.data_list = []
        self.label_dict = {}  # 标签和数值之间的对应关系
        # 获取所有子文件夹的路径
        subfolders = [f.path for f in os.scandir(label_path) if f.is_dir()]
        for folder in subfolders:
            label = os.path.basename(folder)
            if label not in self.label_dict:
                self.label_dict[label] = len(self.label_dict)  # 将新的标签加入字典中
            label_value = self.label_dict[label]  # 获取标签对应的数值
            image_files = glob.glob(os.path.join(folder, '*.jpg'))  # 获取子文件夹中的图像文件
            # 生成二维列表，每个元素包含标签和对应的图像文件绝对路径
            for image_file in image_files:
                self.data_list.append([os.path.abspath(image_file),label_value])
        # 传入定义好的数据处理方法，作为自定义数据集类的一个属性
        self.transform = transform

    def __getitem__(self, index):
        """
        步骤三：实现 __getitem__ 函数，定义指定 index 时如何获取数据，并返回单条数据（样本数据、对应的标签）
        """
        # 根据索引，从列表中取出一个图像
        image_path, label = self.data_list[index]
        # 读取灰度图
        image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1)
        # 飞桨训练时内部数据格式默认为float32，将图像数据格式转换为 float32
        image = image.astype('float32')
        # 应用数据处理方法到图像上
        if self.transform is not None:
            image = self.transform(image)
        # CrossEntropyLoss要求label格式为int，将Label格式转换为 int
        label = int(label)
        # 返回图像和对应标签
        return image, label

    def __len__(self):
        """
        步骤四：实现 __len__ 函数，返回数据集的样本总数
        """
        return len(self.data_list)

# 定义图像归一化处理方法，这里的CHW指图像格式需为 [C通道数，H图像高度，W图像宽度]
transform = Normalize(mean=[127.5], std=[127.5], data_format='CHW')
# 打印数据集样本数
Data = MyDataset(r"./face", transform)
print('样本总数量：',len(Data))
print("样本标签对应码表：\n",Data.label_dict)

# 计算训练集和测试集的划分比例
train_ratio = 0.8
test_ratio = 1 - train_ratio

# 计算划分后的样本数量
train_size = int(train_ratio * len(Data))
test_size = len(Data) - train_size

# 使用paddle.io.random_split函数划分数据集
train_dataset, test_dataset = paddle.io.random_split(Data, [train_size, test_size])

# 打印训练集和测试集的样本数量
print("训练集样本数:", len(train_dataset))
print("测试集样本数:", len(test_dataset))

#
# from paddlex import hub
# # 使用VisualDL可视化训练过程
# logs_train = './output/face_recognize/train'
# logs_val = './output/face_recognize/val'
# hub.visualdl(logs_train=logs_train, logs_val=logs_val, port=8080)


#模型组网，构建并初始化一个模型 mnist
#手写数字识别网络
# mnist = paddle.nn.Sequential(
#     paddle.nn.Flatten(1, -1),
#     paddle.nn.Linear(10000, 512),
#     paddle.nn.ReLU(),
#     paddle.nn.Dropout(0.2),
#     paddle.nn.Linear(512, 13)
#)
#CNN网络构建
CNN=CNN_model_Sequential = nn.Sequential(
    nn.Conv2D(1, 6, (3,3), stride=1),
    nn.ReLU(),
    nn.MaxPool2D(3, 3),
    nn.Conv2D(6, 12, 5, stride=1, padding=0),
    nn.ReLU(),
    nn.MaxPool2D(3, 3),
    nn.Flatten(),
    nn.Linear(972, 120),
    nn.Linear(120, 84),
    nn.Linear(84, 13)
)

#载入模型
CNN_model=paddle.Model(CNN)
CNN_model.prepare(paddle.optimizer.Adam(parameters=CNN_model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())
#模型存放地址
filename = ".\\checkpoint\\test.pdopt"

#是否存在训练好的模型
if os.path.isfile(filename):
    print(f"文件 {filename} 存在于当前目录中")
    user_input = input("是否重新训练：")
    if user_input == "y" or user_input == "Y":
        CNN_model.fit(train_dataset,
                  epochs=4,
                  batch_size=64,
                  verbose=1)
        CNN_model.save('checkpoint/test')
    else:
        CNN_model.load('checkpoint/test')
else:
    print(f"文件 {filename} 不存在于当前目录中")
    #模型训练
    CNN_model.fit(train_dataset,
              epochs=4,
              batch_size=64,
              verbose=1)
    CNN_model.save('checkpoint/test')

print("模型误差分析中......")
eval_result = CNN_model.evaluate(test_dataset, verbose=1)
# 用 predict 在测试集上对模型进行推理
test_result_a = CNN_model.predict(test_dataset)
# 由于模型是单一输出，test_result的形状为[1, 10000]，10000是测试数据集的数据量。这里打印第一个数据的结果，这个数组表示每个数字的预测概率
print("损失（loss）:{}\n准确率（accuracy）：{}".format(eval_result['loss'], eval_result['acc']))
print("测试集总数量：",len(test_result_a[0]))

# 码表
test_dict = {value: key for key, value in Data.label_dict.items()}

while True:
    mode = input("选择你的预测方式：\na:测试集下标预测  b:选择本地文件预测  c:退出\n")
    if mode == "A" or mode == "a":
        # 从测试集中取出一张图片
        a=input('输入你要预测的图片编号：')
        if a=="exit" or a=="退出":
            break
        elif int(a)>=0 and int(a)<=len(test_result_a[0])-1:
            a=int(a)
        else:
            print("输入非法，请重新输入！")
            continue

        #获取图像和标签
        img, label = test_dataset[a]

        # 打印推理结果，这里的argmax函数用于取出预测值中概率最高的一个的下标，作为预测标签
        pred_label = test_result_a[0][a].argmax()
        print('真实结果: {}     预测结果: {}'.format(test_dict[label], test_dict[pred_label]))

        # 使用matplotlib库，可视化图片
        from matplotlib import pyplot as plt
        print(img.shape)
        plt.imshow(img[0],cmap='gray')
        plt.show()
    elif mode == "B" or mode == "b":
        myData = MyDataset1(transform)
        test_result_b=CNN_model.predict(myData)
        for i in range(len(myData)):
            pred_label = test_result_b[0][i].argmax()
            print('图片:{}     预测结果: {}'.format(os.path.basename(myData.data_list[i]),test_dict[pred_label]))
    elif mode == "C" or mode == "c":
        break
    else:
        print("输入非法，请重新输入！")
        continue
pass