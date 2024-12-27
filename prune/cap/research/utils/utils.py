import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_data2(root1: str,root2: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root1), "dataset root: {} does not exist.".format(root1)
    assert os.path.exists(root2), "dataset root: {} does not exist.".format(root2)
    trainclass=root1
    valclass=root2
    # 遍历文件夹，一个文件夹对应一个类别
    train_class = [cla for cla in os.listdir(root1) if os.path.isdir(os.path.join(root1, cla))]
    val_class = [cla for cla in os.listdir(root1) if os.path.isdir(os.path.join(root1, cla))]
    # 排序，保证各平台顺序一致
    train_class.sort()
    val_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(train_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num1 = []  # 存储每个类别的样本总数
    every_class_num2 = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in train_class:
        cla_path = os.path.join(root1, cla)
        # 遍历获取supported支持的所有文件路径
        trainimages = [os.path.join(root1, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        trainimages.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num1.append(len(trainimages))
        for img_path in trainimages:
            train_images_path.append(img_path)
            train_images_label.append(image_class)
    for cla in val_class:
        cla_path = os.path.join(root2, cla)
        # 遍历获取supported支持的所有文件路径
        valimages = [os.path.join(root2, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        valimages.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num2.append(len(valimages))
        for img_path in valimages:
            val_images_path.append(img_path)
            val_images_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num1)+sum(every_class_num2)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
import numpy as np
from sklearn.metrics import classification_report, average_precision_score,roc_auc_score,precision_recall_fscore_support
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    all_labels = []  # 用于存储真实标签的列表
    all_preds = []  # 用于存储预测标签的列表
    all_scores = []  # 用于存储模型对所有类别的预测概率
    all_scores2 = []
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        all_labels.extend(labels.cpu().numpy())  # 将真实标签添加到列表
        all_preds.extend(pred_classes.cpu().numpy())  # 将预测标签添加到列表
        all_scores.extend(torch.nn.functional.softmax(pred, dim=1).cpu().numpy())  # 将预测概率添加到列表
        all_scores2.extend(torch.nn.functional.softmax(pred, dim=1)[:,1].cpu().numpy())
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    # 将 all_scores 转换为2D数组
    #all_scores_2d = np.array(all_scores).reshape(-1, 1)
    print(len(all_labels))
    print(np.array(all_scores).shape)
    if np.array(all_scores).shape[1]==2:
        auc = roc_auc_score(all_labels, all_scores2, average=None, multi_class='ovr')
        ap_score = average_precision_score(all_labels, all_scores2, average=None)
    else:
        auc = roc_auc_score(all_labels, all_scores, average=None, multi_class='ovr')
        ap_score = average_precision_score(all_labels, all_scores, average=None)

    prec,recall,f1,_=precision_recall_fscore_support(all_labels, all_preds,zero_division=0)
    cls_rpt = classification_report(all_labels, all_preds, output_dict=True)
    '''cls_rpt=classification_report(all_labels, all_preds, output_dict=True)
    f1_score_macro = cls_rpt['macro avg']['f1-score']
    f1_score_weighted=cls_rpt['weighted avg']['f1-score']  
    pre_macro = cls_rpt['macro avg']['precision']
    pre_weighted = cls_rpt['weighted avg']['precision']
    recall_macro = cls_rpt['macro avg']['recall']
    recall_weighted = cls_rpt['weighted avg']['recall']'''
    #accuracy_per_class = classification_report(all_labels, all_preds, output_dict=True)['accuracy']
    #print(cls_rpt)
    print("map:{:.3f}".format(ap_score.mean()))
    #print("f1_score:{:.3f} accuracy_per_class:{:.3f} ",f1_score,accuracy_per_class)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,auc.mean(),f1.mean(),prec.mean(),recall.mean(),ap_score.mean()