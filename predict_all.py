#!/usr/bin/python  
# -*- coding:utf8 -*-

import predict
import os

allFileNum = 0  
def printPath(path, level = 1):  
    global allFileNum  
    ''''' 
    打印一个目录下的所有文件夹和文件 
    level:当前目录级别
    path：当前目录路径
    '''  
    # 次目录的级别  
    dirList = []  
    # 所有文件  
    fileList = []  
    # 返回一个列表，其中包含在目录条目的名称 
    files = os.listdir(path)  
    # 先添加目录级别  
    dirList.append(str(level))  
    for f in files:  
        if(os.path.isdir(path + '/' + f)):  
            # 排除隐藏文件夹。因为隐藏文件夹过多  
            if(f[0] == '.'):  
                pass  
            else:  
                # 添加非隐藏文件夹  
                dirList.append(f)  
        if(os.path.isfile(path + '/' + f)):  
            # 添加文件  
            fileList.append(f)  
    # 当一个标志使用，文件夹列表第一个级别不打印  
    i_dl = 0  
    for dl in dirList:  
        if(i_dl == 0):  
            i_dl = i_dl + 1  
        else:  
            # 打印至控制台，不是第一个的目录  
            print('-' * (int(dirList[0])), dl) 
            # 打印目录下的所有文件夹和文件，目录级别+1  
            printPath(path + '/' + dl, (int(dirList[0]) + 1))  
    for fl in fileList:  
        # 打印文件
        if ".jpg" in fl or ".png" in fl or ".JPG" in fl or ".PNG" in fl:
            # 预测此文件在对应网络中的识别率
            current_path = path +'/' + fl
            print("-" * (int(dirList[0])), fl)
            predict.predict_by_file(input_image_path = current_path,checkpoint_file_path='checkpoint_dir\\resnet50_102improve8.pth')
        # 随便计算一下有多少个文件  
        allFileNum = allFileNum + 1
  
if __name__ == '__main__':
    printPath(path='D:\Desktop\部分识别样本\部分识别样本')
    print('总文件数 =', allFileNum)
    