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
    parsed_data_path = r'C:\Users\XPS\Desktop\数据\增广的数据\sequence.npz'
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
    model = TrainingModel_RNNVec(len(word2ix_train), len(word2ix_fix), 200, 400)
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
        testresult = generate_test(model, testinput, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix,opt)
        #print(testresult)
        inornot, rankingnum = final_test(testresult,testtarget)

        loss_meter_inornot.add(inornot)
        loss_meter_rankingnum.add(rankingnum)
        print("inornot(TE): " + str(loss_meter_inornot.value()[0]))
        print("rankingnum(SR): " + str(loss_meter_rankingnum.value()[0]))

        model.train() 
    t.save(model.state_dict(), '%s_%s.pth' % ("testforrnn_arg", epoch))



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
    model = TrainingModel_RNNVec(len(word2ix_train), len(word2ix_fix), 200, 400)
    map_location = lambda s, l: s
    state_dict = t.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    pretrained_weight = form_matrix(ix2word_fix,opt.pathforvec)
    pretrained_weight = np.array(pretrained_weight)
    model.embeddingsfix.weight.data.copy_(t.from_numpy(pretrained_weight))
    

    if opt.use_gpu:
        model.cuda()

    result = generate(model, stripresult, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix,opt)
    # write the result into the csv
    write_csv(result,opt.writepath)
    ency,lossprecent = covert_to_encrypt(result,opt.tablepath,opt.departmentpath)
    write_csv(ency,opt.writepath_ency)
    print(lossprecent)


# the called function for generation with the vector for the key word
def gen_with_vector(**kwargs):
    #入口数据就为两个 一个是部门 一个是关键词向量
    
    for k, v in kwargs.items():
        setattr(opt, k, v.strip("'"))

    file_object=open(opt.testsetpath,'r',encoding='utf-8')
    data = file_object.read()
    file_object.close()
    data = data.split('\n')
    finaldata = data_split_inner(data,' ')

    #pair = [['公安',[0.185774,0.240292,0.004667,0.194717,-0.602632,0.245656,0.258343,-0.163758,-0.105399,-0.483415,-0.119850,-0.061783,0.141093,-0.278988,-0.238508,-0.270287,-0.363062,-0.526103,0.324706,0.214232,-0.444421,-0.389982,-0.208329,-0.250402,0.140813,0.612041,-0.388810,0.442441,0.094806,0.403121,-0.075520,-0.215281,0.472267,0.539537,-0.198181,0.521075,0.025065,-0.073142,-0.005518,-0.466331,-0.011482,0.455457,0.522161,-0.429649,-0.167342,0.172249,-0.498246,0.040961,0.166752,0.165913,-0.046194,0.102941,0.282750,-0.111863,0.200259,-0.473747,0.165709,0.321475,-0.060773,0.056295,-0.124481,0.078447,0.245033,0.201244,-0.220066,-0.473854,0.156142,0.351557,-0.077365,-0.418671,-0.576947,-0.327172,-0.059312,0.141462,-0.138260,0.015616,0.242208,0.100440,-0.126658,-0.344706,0.078268,0.353345,-0.459928,0.355435,-0.298799,-0.584326,-0.035596,-0.056843,0.008961,-0.737625,0.650411,0.302178,-0.308775,-0.074439,-0.005482,-0.331152,-0.406975,0.050284,0.081398,0.085459,-0.102804,0.044128,-0.430915,-0.011778,-0.427645,0.239175,-0.032082,-0.560575,-0.344889,0.369777,-0.258999,-0.389398,-0.131652,-0.104312,-0.404019,0.167957,-0.160930,-0.291415,0.517506,-0.243406,0.020856,-0.411876,-0.346382,0.143864,-0.058643,0.134075,-0.060899,0.193994,-0.480253,-0.195371,-0.381812,-0.214213,-0.123320,0.447235,-0.057068,-0.235316,-0.311169,0.095653,-0.045980,-0.039832,-0.087425,0.001695,-0.472713,-0.641947,0.081975,0.175762,0.199911,0.501622,0.092125,0.014051,0.233818,-0.270307,-0.202669,0.033757,-0.427492,-0.006059,0.611274,-0.066681,0.512747,-0.095007,0.423356,0.128483,-0.380595,0.004818,0.691321,-0.063294,-0.171382,0.145801,0.376692,-0.209230,0.070547,-0.601059,-0.304845,0.242572,-0.136051,-0.039179,0.377138,0.041296,-0.105958,-0.076830,0.236428,-0.347921,-0.091339,0.352473,0.561357,-0.127522,0.266830,0.373282,0.559688,-0.023055,0.685433,0.006953,0.516254,0.327722,0.525244,0.392855,-0.256328,0.374918,-0.022773,0.486938]]]
    #读vector
    vec = read_vec(opt.pathfortestvec)
    #换成vector
    pair = converttovector(finaldata,vec)

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

    result = generate_with_vector(model, pair, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix,opt)
    #print(result[0][0])
    #print(result[0][2:])
    result = rever_to_name(result,vec)
    #print(result)

    #分别写csv和txt
    write_csv(result,opt.writepath)
    ency,lossprecent = covert_to_encrypt(result,opt.tablepath,opt.departmentpath)
    write_csv(ency,opt.writepath_ency)
    print(lossprecent)


if __name__ == '__main__':
    import fire

    fire.Fire()

