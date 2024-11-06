from tqdm import tqdm
import torch
import gc 


def train_val(model, loader_train, loader_val, opt, criterion, epoch):
    # 初始化用于存储训练和验证损失及准确率的列表
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    # 迭代进行指定数量的训练周期（epochs）
    for e in range(epoch):
        # 调用训练函数，获取当前周期的训练损失和准确率
        loss_t, acc_t = train(e, model, loader_train, opt, criterion)
        # 将当前周期的训练损失和准确率添加到列表中
        train_loss.append(loss_t)
        train_acc.append(acc_t)

        # 调用验证函数，获取当前周期的验证损失和准确率
        loss_v, acc_v = validation(e, model, loader_val, criterion)
        # 将当前周期的验证损失和准确率添加到列表中
        val_loss.append(loss_v)
        val_acc.append(acc_v)

        # 进行垃圾回收以释放内存
        gc.collect()
        
        # 清空 GPU 缓存以释放内存
        torch.cuda.empty_cache()

    # 返回训练后的模型和记录的训练及验证损失和准确率
    return model, train_loss, train_acc, val_loss, val_acc


def train(e, model, loader, opt, criterion):
    model.train()  # 将模型设置为训练模式
    treshold = 0.5  # 设置分类的阈值为0.5
    acc_loss = 0  # 初始化累计损失
    correct = 0  # 初始化正确分类的数量
    target_sum = 0  # 初始化目标样本的总数

    tqdm_loader = tqdm(loader)  # 如果需要可视化进度条，可以解开注释

    # 遍历数据加载器
    for i, (img, target) in enumerate(tqdm_loader):
        img = img.float()  # 将图像数据转换为浮点型
        # img = img.permute(0, 3, 1, 2).float()  # 如果需要调整图像维度，可以解开注释
        target = target.float()  # 将目标标签转换为浮点型

        img = img.cuda()  # 将图像数据移至GPU
        target = target.cuda()  # 将目标标签移至GPU

        opt.zero_grad()  # 清除之前的梯度
        feature, x = model(img)
        output = torch.sigmoid(x).float()
        #output = torch.sigmoid(model(img)).float()  # 进行前向传播并通过sigmoid激活函数获得预测概率
        loss = criterion(output, target) #+ 0.1* model.module.compute_arcface_loss(feature, target) # 计算损失

        loss.backward()  # 反向传播计算梯度
        opt.step()  # 更新模型参数

        acc_loss += loss.item()  # 累加损失
        avg_loss = acc_loss / (i + 1)  # 计算当前平均损失

        output = torch.where(output > treshold, 1, 0)  # 将预测概率转换为二进制输出

        # correct += output.eq(target.view_as(output)).sum().item() / (6 * len(target))  # 计算正确率的另一种方式，注释掉

        res = output == target  # 判断预测是否与真实标签相等

        # 统计正确分类的样本数量
        for tensor in res:
            if False in tensor:
                continue  # 如果某一行中有误分类，则跳过
            else:
                correct += 1  # 所有类别都正确，则正确计数加一

        target_sum += len(target)  # 更新目标样本总数
        avg_acc = correct / (target_sum)  # 计算当前的平均准确率

        # 更新进度条描述，确保没有注释掉
        tqdm_loader.set_description("Epoch {}, train_loss={:4} , acc={:4}".format(e, round(avg_loss, 4), round(avg_acc, 4)))  # 更新进度条描述，注释掉

        del img  # 删除图像变量以释放内存
        del target  # 删除目标变量以释放内存
        del output  # 删除输出变量以释放内存
        del loss  # 删除损失变量以释放内存
        del res  # 删除结果变量以释放内存

        gc.collect()  # 垃圾回收，释放未使用的内存
        torch.cuda.empty_cache()  # 清空GPU缓存

    # 打印每个epoch的平均损失和准确率
    print("Epoch {}, train_loss={:4} , acc={:4}".format(e, round(avg_loss, 4), round(avg_acc, 4)))
    
    del acc_loss  # 删除累计损失以释放内存
    del target_sum  # 删除目标总数以释放内存
    del correct  # 删除正确计数以释放内存

    return avg_loss, avg_acc  # 返回平均损失和准确率



def validation(e, model, loader, criterion):
    model.eval()  # 将模型设置为评估模式，这样会禁用Dropout等特性
    tqdm_loader = tqdm(loader)  # 可视化进度条
    treshold = 0.5  # 设置分类的阈值为0.5
    acc_loss = 0  # 初始化累计损失
    correct = 0  # 初始化正确分类的数量
    target_sum = 0  # 初始化目标样本的总数

    # 遍历数据加载器
    for i, (img, target) in enumerate(tqdm_loader):  # 使用 tqdm_loader
        img = img.float()  # 将图像数据转换为浮点型
        # img = img.permute(0, 3, 1, 2).float()  # 如果需要调整图像维度，可以解开注释
        target = target.float()  # 将目标标签转换为浮点型

        img = img.cuda()  # 将图像数据移至GPU
        target = target.cuda()  # 将目标标签移至GPU

        with torch.no_grad():  # 在评估模式下不计算梯度
            _, x = model(img)
            output = torch.sigmoid(x).float()
            #output = torch.sigmoid(model(img)).float()  # 进行前向传播并通过sigmoid激活函数获得预测概率
            loss = criterion(output, target)  # 计算损失
        
        acc_loss += loss.item()  # 累加损失
        avg_loss = acc_loss / (i + 1)  # 计算当前平均损失
        
        output = torch.where(output > treshold, 1, 0)  # 将预测概率转换为二进制输出
        
        # correct += output.eq(target.view_as(output)).sum().item() / (6 * len(target))  # 计算正确率的另一种方式，注释掉

        res = output == target  # 判断预测是否与真实标签相等
        
        # 统计正确分类的样本数量
        for tensor in res:
            if False in tensor:
                continue  # 如果某一行中有误分类，则跳过
            else:
                correct += 1  # 所有类别都正确，则正确计数加一
        
        target_sum += len(target)  # 更新目标样本总数
        avg_acc = correct / (target_sum)  # 计算当前的平均准确率
        
        # 更新进度条描述
        tqdm_loader.set_description("Epoch {}, val_loss={:4} , val_acc={:4}".format(e, round(avg_loss, 4), round(avg_acc, 4)))

        del img  # 删除图像变量以释放内存
        del target  # 删除目标变量以释放内存
        del output  # 删除输出变量以释放内存
        del res  # 删除结果变量以释放内存
        
        gc.collect()  # 垃圾回收，释放未使用的内存
        torch.cuda.empty_cache()  # 清空GPU缓存

    # 打印每个epoch的平均损失和准确率
    print("Epoch {}, val_loss={:4} , acc={:4}".format(e, round(avg_loss, 4), round(avg_acc, 4)))

    del target_sum  # 删除目标总数以释放内存
    del acc_loss  # 删除累计损失以释放内存
    del correct  # 删除正确计数以释放内存

    return avg_loss, avg_acc  # 返回平均损失和准确率
