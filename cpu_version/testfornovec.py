# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os
import torch as t
from torch import nn
import tqdm
from torchnet import meter
import ipdb
from torch.autograd import Variable
from utils import Visualizer
import random
import csv
from model import *
from function import *


#config the parameter
class Config(object):
	use_gpu = False
	env = 'mytest'  # visdom env
	parsed_data_path = r'C:\Users\XPS\Desktop\数据\没有增广的数据\sequence_without_vec_without_arg.npz'
	batch_size = 128
	model_path = r"C:\Users\XPS\Desktop\test_19_28.pth" # 预训练模型路径
	epoch = 20
	plot_every = 10  # 每20个batch 可视化一次
	debug_file = '/tmp/debugp'
	lr = 1e-3
	start_words = ['法院','工商登记号']  # 诗歌开始

opt = Config()

#function for training
def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v.strip("'"))

    opt.device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    device = opt.device
    vis = Visualizer(env=opt.env)

    # 获取数据
    data, word2ix, ix2word = load_data_(opt.parsed_data_path)


    data = t.from_numpy(data)
    print("datalen:" + str(len(data)))
    random.shuffle(data)

    devision = int(len(data)*8/10)
    train_data = data[:devision]
    test_data = data[devision + 1:]


    dataloader = t.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    dataloader_fortest = t.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    # 模型定义
    model = TrainingModel(len(word2ix), 200, 400)
    optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    #预训练模型
    '''
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    model.to(device)'''
    model.to(device)

    loss_meter = meter.AverageValueMeter()
    i = 0

    for epoch in range(opt.epoch):
        loss_meter.reset()
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]#后面这个是是去掉了第一行，前面这个是去掉最后一行
            #print('\n')
            #print('\n')
            #print(type(input_))
            '''
            if i == 0:
	            print(transfer_and_print_(ix2word,input_.numpy()))
	            print(transfer_and_print_(ix2word,target.numpy()))
	            i +=1'''
            #print('input_:\n' + str(input_.numpy()))
            #print('target:\n' + str(target.numpy()))
            #print(input_.size())
            #print(data_.size())
            #print('data_:\n' + str(data_))
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            # 可视化
            if (1 + ii) % opt.plot_every == 0:

                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                vis.plot('loss', loss_meter.value()[0])
                vis.plot('loss', loss.item())

                
                poetrys = [[ix2word[_word] for _word in data_[:, _iii].tolist()]
                           for _iii in range(data_.shape[1])][:16]

        # for test
        loss_meter.reset()
        model.eval()      # 设置为test模式  
        test_loss = 0     # 初始化测试损失值为0
        correct = 0       # 初始化预测正确的数据个数为0
        total = 0
        for iii, datatest in enumerate(dataloader_fortest):
            #if args.cuda:
            #   data, target = data.cuda(), target.cuda()
            #print(datatest)
            datatest = datatest.long().transpose(1, 0).contiguous()
            datatest = datatest.to(device)
            optimizer.zero_grad()
            input_test, target_test = datatest[:-1, :], datatest[1:, :]#后面这个是是去掉了第一行，前面这个是去掉最后一行
            output_test, _ = model(input_test)
            test_loss += criterion(output_test, target_test.view(-1))
            #print("loss_test: " + str(loss_test))
            #loss_meter.add(loss_test.item())
            #test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss 把所有loss值进行累加
            pred = output_test.data.max(1, keepdim=True)[1] # get the index of the max log-probability 其中[0]是值[1]是index
            #print(output_test.size())
            #print(target_test.size())
            #print("right: " + str(pred.eq(target_test.data.view_as(pred)).cpu().sum()))
            #print(pred.size()[0])
            #print(target_test)
            target_test = target_test.data.view_as(pred)[int(pred.size()[0]/4*2) : int(pred.size()[0]/4*3)]
            #print(target_test)
            pred = pred[int(pred.size()[0]/4*2) : int(pred.size()[0]/4*3)]

            correct += pred.eq(target_test).cpu().sum()  # 对预测正确的数据个数进行累加
            total += target_test.size()[0]
            #correct += find_in_ten(output_test.data,target_test.data)

        test_loss /= iii
        print(epoch)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss , correct, total,
         100. * correct / total))

    t.save(model.state_dict(), '%s_%s.pth' % ("test_witoutvec", epoch))


# called function for generation with index with multi pair
def multi_gen(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v.strip("'"))

    file_object=open(opt.testsetpath,'r',encoding='utf-8')
    data = file_object.read()
    file_object.close()
    data = data.split('\n')
    finaldata = data_split_inner(data,',')


    #先找出三元组内前两个元素
    stripresult = mystrip(finaldata,len(finaldata[1]))
    #print(opt.parsed_data_path)

    data, word2ix, ix2word = load_data_(opt.parsed_data_path)
    model = TrainingModel(len(word2ix), 200, 400)
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()

    result = generate_sim(model, stripresult, ix2word, word2ix)
    #print(','.join(result))

    write_csv(result,opt.writepath)
    ency,lossprecent = covert_to_encrypt(result,opt.tablepath,opt.departmentpath)
    write_csv(ency,opt.writepath_ency)
    print(lossprecent)

if __name__ == '__main__':
    import fire

    fire.Fire()

