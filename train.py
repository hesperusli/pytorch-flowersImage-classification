import torch
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import os
from torch import nn,optim
import config
import Models
import datasets
from torch.autograd import Variable
from utils.utils import *   #导入utils的所有函数和变量
from test import *
from torch.autograd import Variable

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

if __name__ == '__main__':
    # 检查文件夹是否存在，不存在就创建
    if not os.path.exists(config.example_folder):
        os.mkdir(config.example_folder)
    if not os.path.exists(config.weights):
        os.mkdir(config.weights)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)
    #定义模型
    model=Models.getNet()
    if torch.cuda.is_available():
        model=model.cuda()
    # print(model)
    #定义优化器
    optimizer=optim.SGD(model.parameters(),lr=config.lr,weight_decay=config.lr_decay,momentum=0.9)
    # 定义损失函数
    loss_criterion=nn.CrossEntropyLoss().to(device)
    # 检查是否需要加载checkpoint已经训练好的模型训练
    start_epoch=0
    current_accuracy=0
    resume= False  # 默认不加载
    if resume:
        checkpoint=torch.load(config.weights+config.model+'.pth')
        start_epoch=checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    # 定义训练集和测试集
    input_data=datasets.train_data
    train_loader=DataLoader(input_data,batch_size=config.bach_size,shuffle=True,num_workers=0)
    test_loader=DataLoader(datasets.test_data,batch_size=config.bach_size,shuffle=False,num_workers=0)
    print(train_loader)

    # 开始训练
    train_loss=[]
    acc=[]
    test_loss=[]
    print("-------------------------------start Training----------------------------------")
    for epoch in range(start_epoch,config.epochs):
        model.train()
        config.lr=lr_step(epoch)
        optimizer=optim.SGD(model.parameters(),lr=config.lr,momentum=0.9,weight_decay=config.weight_decay)
        loss_epoch=0
        for index,(input,target) in enumerate(train_loader):
            model.train()
            input=Variable(input).to(device)
            target=Variable(torch.from_numpy(np.array(target)).long().to(device))
            #梯度清零
            optimizer.zero_grad()
            output=model(input)
            loss=loss_criterion(output,target)
            #反向传播和参数更新
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()*input.size(0)
            if (index+1) % 10 == 0:
                print("Epoch: {} [{:>3d}/{}]\t Loss: {:.6f} ".format(epoch+1,index*config.bach_size,
                                                                  len(train_loader.dataset),loss.item()))
        loss_epoch=loss_epoch/len(train_loader.dataset)
        print("loss:",loss_epoch)
        if (epoch+1)%1 == 0:
            print("-------------------------------Eavluate----------------------------------")
            model.eval()
            # 测试的话使用测试集
            test_loss1,accTop1=evaluate(test_loader,model,loss_criterion)
            acc.append(accTop1)
            print("type(accTop1) =",type(accTop1))
            print(accTop1)
            test_loss.append(test_loss1)
            train_loss.append(loss_epoch/len(train_loader))
            print("Test_epoch: {} Test_accurary: {:.4} Test_loss: {:.6f}".format(epoch+1,accTop1,test_loss1))
            save_model=accTop1>current_accuracy
            accTop1=max(current_accuracy,accTop1)
            current_accuracy=accTop1
            save_checkpoint({
                "epoch":epoch+1,
                "model_name":config.model,
                "state_dict":model.state_dict(),
                "accTop1":current_accuracy,
                "optimizer":optimizer.state_dict(),
            },save_model)

