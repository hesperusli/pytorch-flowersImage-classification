import torch
import sklearn.metrics

device=torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
#评估模型

def evaluate(test_loader, model, criterion):
  # set the model to evaluation mode
  model.eval()

  # initialize the running loss and accuracy
  running_loss = 0.0
  running_acc = 0.0

  # loop through the test data loader
  for inputs, targets in test_loader:
    # move the inputs and targets to the device
    inputs = inputs.to(device)
    targets = targets.to(device)

    # forward pass and get the output logits
    outputs = model(inputs)
    # compute the loss
    loss = criterion(outputs, targets)

    # get the predictions
    _, preds = torch.max(outputs, 1)
    # compute the accuracy
    acc = sklearn.metrics.accuracy_score(targets.cpu(), preds.cpu())

    # update the running loss and accuracy
    running_loss += loss.item() * inputs.size(0)
    running_acc += acc * inputs.size(0)

  # calculate the average loss and accuracy
  avg_loss = running_loss / len(test_loader.dataset)
  avg_acc = running_acc / len(test_loader.dataset)

  return avg_loss, avg_acc
# def evaluate(test_loader, model, criterion):
#   # 设置模型为评估模式
#   model.eval()
#   # 初始化测试损失和正确预测的数量
#   test_loss = 0
#   correct = 0
#   # 遍历数据加载器中的所有批次
#   with torch.no_grad(): # 不计算梯度，节省内存
#     for inputs, labels in test_loader:
#       # 把输入数据和标签移动到指定的设备上
#       inputs = inputs.to(device)
#       labels = labels.to(device)
#       # 前向传播，得到模型的输出
#       outputs = model(inputs)
#       # 计算损失，并累加到测试损失上
#       loss = criterion(outputs, labels)
#       test_loss += loss.item()
#       # 得到预测的类别，即输出中最大值的索引
#       _, predicted = torch.max(outputs, 1)
#       # 更新正确预测的数量
#       correct += (predicted == labels).sum().item()
#   # 计算平均测试损失，即每个样本的损失
#   test_loss /= len(test_loader.dataset)
#   # 计算准确率，即正确预测的比例
#   accuracy = correct / len(test_loader.dataset)
#   # 返回测试损失和准确率
#   return test_loss, accuracy