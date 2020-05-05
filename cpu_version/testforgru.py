# -*- coding: utf-8 -*-
import numpy as np
import sys, os
import torch as t
from torch import nn
import torch.nn.functional as F
from torchnet import meter
import tqdm
import ipdb
from torch.autograd import Variable
from utils import Visualizer
import random
import heapq
import csv
from model import *
from function import *


# for seting the parameter and the path
class Config(object):
    use_gpu = False
    env = 'mytestforvector'  # visdom env
    parsed_data_path = r'C:\Users\XPS\Desktop\数据\没有增广的数据\sequence_without_arg.npz'
    batch_size = 128
    model_path = r"C:\Users\XPS\Desktop\训练结果\28\testtestingfix_19_withoutarg_28.pth" # 预训练模型路径
    epoch = 20
    plot_every = 20  # 每20个batch 可视化一次
    debug_file = '/tmp/debugp'
    lr = 1e-3
    pathforvec = r'C:\Users\XPS\Desktop\数据\增广的数据\vec.txt'


opt = Config()


#function for training
def train(**kwargs):
    # setting the parameter in opt as the input argument
    for k, v in kwargs.items():
        setattr(opt, k, v.strip("'"))
    loss_meter_inornot = meter.AverageValueMeter()
    loss_meter_rankingnum = meter.AverageValueMeter()
    # setting the device
    opt.device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    device = opt.device
    #vis = Visualizer(env=opt.env)

    # get the sequence from sequence.npz
    data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix = load_data(opt.parsed_data_path)

    random.shuffle(data)
    
    #devide the data for the test and train and convert to the dataloader
    devision = int(len(data)*8/10)
    train_data = data[:devision]
    test_data = data[devision + 1:]
    train_data = t.from_numpy(train_data)
    test_data = t.from_numpy(test_data)
    dataloader = t.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    dataloader_fortest = t.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    # define the model
    model = TrainingModel_GRUVec(len(word2ix_train), len(word2ix_fix), 200, 400)
    optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()

    #loss_meter = meter.AverageValueMeter()

    # load the pretrained word vector and convert it to a matrix in the order of index
    pretrained_weight = form_matrix(ix2word_fix,opt.pathforvec)
    pretrained_weight = np.array(pretrained_weight)
    # copy the pretrained vectors to the embeding
    model.embeddingsfix.weight.data.copy_(t.from_numpy(pretrained_weight))

    i = 0

    for epoch in range(opt.epoch):
        #loss_meter.reset()
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):

            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]

            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
        # for test
        model.eval()      # 设置为test模式  
        test_loss = 0     # 初始化测试损失值为0
        correct = 0       # 初始化预测正确的数据个数为0
        total = 0
        for iii, datatest in enumerate(dataloader_fortest):
            #if args.cuda:
            #   data, target = data.cuda(), target.cuda()

            datatest = datatest.long().transpose(1, 0).contiguous()
            datatest = datatest.to(device)
            optimizer.zero_grad()
            input_test, target_test = datatest[:-1, :], datatest[1:, :]#后面这个是是去掉了第一行，前面这个是去掉最后一行
            #print(type(input_test))
            output_test, _ = model(input_test)
            test_loss += criterion(output_test, target_test.view(-1))
            #print("loss_test: " + str(loss_test))
            #loss_meter_inornot.add(loss_test.item())
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
            

            #print("original: " + str(len(datatest.data[0])))
            #print(target_test.data.view_as(pred).size()[0])
            #print(target_test.data.view_as(pred).size())
            correct += pred.eq(target_test).sum()  # 对预测正确的数据个数进行累加
            total += target_test.size()[0]
            #correct += find_in_ten(output_test.data,target_test.data)

        test_loss /= iii
        print(epoch)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss , correct, total,
         100. * correct / total))

        testinput, testtarget = data_[1:3, :], data_[3, :]
        #print(testinput)
        testinput=np.transpose(testinput.numpy()).tolist()
        testtarget=np.transpose(testtarget.numpy()).tolist()
        #print(testinput)
        #print(testtarget)
        testresult = generate_test(model, testinput, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix)
        #print(testresult)
        inornot, rankingnum = final_test(testresult,testtarget)

        loss_meter_inornot.add(inornot)
        loss_meter_rankingnum.add(rankingnum)
        print("inornot(TE): " + str(loss_meter_inornot.value()[0]))
        print("rankingnum(SR): " + str(loss_meter_rankingnum.value()[0]))

        model.train() 
    t.save(model.state_dict(), '%s_%s.pth' % ("testforgru_arg", epoch))




# the called function for genernation with index
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
    #print(stripresult)


    data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix = load_data(opt.parsed_data_path)
    model = TrainingModel_GRUVec(len(word2ix_train), len(word2ix_fix), 200, 400)
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    pretrained_weight = form_matrix(ix2word_fix,opt.pathforvec)
    pretrained_weight = np.array(pretrained_weight)
    model.embeddingsfix.weight.data.copy_(t.from_numpy(pretrained_weight))
    

    if opt.use_gpu:
        model.cuda()

    result = generate(model, stripresult, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix)
    # write the result into the csv
    write_csv(result,opt.writepath)
    ency,lossprecent = covert_to_encrypt(result,opt.tablepath,opt.departmentpath)
    write_csv(ency,opt.writepath_ency)
    print(lossprecent)


if __name__ == '__main__':
    import fire

    fire.Fire()

