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
#import ipdb
from torch.autograd import Variable
#from utils import Visualizer
import random


#filepath = r'C:\Users\XPS\Desktop\sequence.txt'
#tablepath = r'C:\Users\XPS\Desktop\唯一表格名字.txt'

class TrainingModel_GRUVec(nn.Module):
    #vocab_size mean how many distinct word would appear in the poem
    def __init__(self, vocab_size_train, vocab_size_fix, embedding_dim, hidden_dim):
        super(TrainingModel_GRUVec, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size_train, embedding_dim)
        self.gru = nn.GRU(embedding_dim, self.hidden_dim, num_layers=2)#lstm里面有参数可以通过梯度下降惊醒拟合，通过input得到输入和输出然后计算loss返回提反独下降
        
        #the embeding for the pretrained word vector
        self.embeddingsfix = nn.Embedding(vocab_size_fix, embedding_dim)
        for p in self.embeddingsfix.parameters():
            p.requires_grad = False
        #self.lstmfix = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)

        self.linear1 = nn.Linear(self.hidden_dim, vocab_size_fix + vocab_size_train)

    def forward(self, input, hidden=None, mark=None, num=None):
        #for training
        if mark == None:
            #the input usuassly is a two demension tensor the size is [seq_len,batchsize]
            #every row of the input is the first element of all the seqences in the this batch
            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                #h_0, c_0 = hidden
                h_0 = hidden

            # size: (seq_len,batch_size,embeding_dim) 去掉tensor的第三行
            # our input have specific format, ["<START>"] + department + keyword + tablename + ["<EOP>"]
            # for the training data the format is : ["<START>"] + department + keyword + tablename 
           
            # size: (seq_len,batch_size,embeding_dim)
            #change for the department and the tablename
            embedschange = self.embeddings(t.cat([input[0:2],input[3:]],dim = 0))
           #change for the keyword
            embedsfix = self.embeddingsfix(input[2:3])
            #concatenate the embedding
            embeds=t.cat([embedschange[0:2],embedsfix,embedschange[2:]],dim = 0)

            
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with index
        elif mark == 1:

            seq_len, batch_size = input.size()
            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
               # h_0, c_0 = hidden
               h_0 = hidden

            # size: (seq_len,batch_size,embeding_dim)
            #for converting the department or table index to vector
            if num == 0:    
                embeds = self.embeddings(input)
            #for converting the keyword index to vector
            else:
                embeds = self.embeddingsfix(input)
            # output size: (seq_len,batch_size,hidden_dim)
            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(seq_len * batch_size, -1))
            return output, hidden
        #for generation with vector
        elif mark == 2:

            #the input is the vector representation of the key word, a 200 demension vector
            embeds = t.tensor(input)

            if hidden is None:
                h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
                #c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            else:
                #h_0, c_0 = hidden
                 h_0  = hidden

            output, hidden = self.gru(embeds, h_0)

            # size: (seq_len*batch_size,vocab_size)
            output = self.linear1(output.view(1, -1))
            return output, hidden


class Config(object):
	use_gpu = True
	env = 'mytestforvector'  # visdom env
	#parsed_data_path = r'C:\Users\XPS\Desktop\sequence.npz'
	parsed_data_path=r'/home/liuzhm/gpu_version/sequence.npz'
	batch_size = 128
	#model_path = "testtestingfix_arg_19_82.pth" # 预训练模型路径
	epoch = 20
	plot_every = 20  # 每20个batch 可视化一次
	debug_file = '/tmp/debugp'
	lr = 1e-3
	#start_words = ['法院','工商登记号']  # 诗歌开始

opt = Config()

def load_data(parsed_data_path):
	data = np.load(parsed_data_path)
	data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix = data['data'], data['word2ix_train'].item(), data['ix2word_train'].item(), data['word2ix_fix'].item(), data['ix2word_fix'].item()
	return data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix

def transfer_and_print(ix2word_train,ix2word_fix,data,fix):
	final = []
	for i,row in enumerate(data):
		print(row)
		temp = []
		for item in row:
			if i == fix:
				temp.append(ix2word_fix[item])
			else:
				temp.append(ix2word_train[item])
		final.append(temp)
	return final


def form_matrix(ix2word_fix):
	path = r"/home/liuzhm/gpu_version/vec.txt"
	targetfile = open(path,'r', encoding='UTF-8')
	target = targetfile.read()
	target = target.split('\n')
	targetfile.close()

	vec = []
	for ele in target:
		temp = ele.split(':')
		name = temp[0]
		realrec = list(map(float,temp[1].split(',')))
		vec.append([name,realrec])

	matrixforfix = []
	for i in range(len(ix2word_fix)):
		word = ix2word_fix[i]
		#print(word)
		for item in vec:
			if word == item[0]:
				matrixforfix.append(item[1])

	return matrixforfix



def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v.strip("'"))
    opt.device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    device = opt.device
    loss_meter_inornot = meter.AverageValueMeter()
    loss_meter_rankingnum = meter.AverageValueMeter()
    #vis = Visualizer(env=opt.env)

    # 获取已经生成好的序列sequence.npz
    data, word2ix_train, ix2word_train, word2ix_fix, ix2word_fix = load_data(opt.parsed_data_path)

    random.shuffle(data)

    devision = int(len(data)*8/10)
    train_data = data[:devision]
    test_data_origin = data[devision + 1:]

    #data, word2ix, ix2word = load_data(opt.parsed_data_path)
    train_data = t.from_numpy(train_data)
    test_data = t.from_numpy(test_data_origin)
    dataloader = t.utils.data.DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)
    dataloader_fortest = t.utils.data.DataLoader(test_data, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    # 模型定义
    model = TrainingModel_GRUVec(len(word2ix_train), len(word2ix_fix), 200, 400)

    #optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
    optimizer = t.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()


    #加载训练好的词向量
    #首先读入向量数据
    #拼接成矩阵
    #转换并输入
    pretrained_weight = form_matrix(ix2word_fix)
    pretrained_weight = np.array(pretrained_weight)
    model.embeddingsfix.weight.data.copy_(t.from_numpy(pretrained_weight))
    
    #训练前的embedding
    #hello_idx = torch.LongTensor([word2ix_fix["最后更新时间"]])
    #hello_idx = Variable(hello_idx)
    #hello_embed = model.embeddingsfix(hello_idx)
    #print("before train: \n")
    #print(hello_embed)	
    #if opt.model_path:
    #    model.load_state_dict(t.load(opt.model_path))
    model.to(device)

    i = 0

    for epoch in range(opt.epoch):
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):

            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]#后面这个是是去掉了第一行，前面这个是去掉最后一行
            #print('\n')
            #print('\n')
            #print(type(input_))
            
            #if i == 0:
	            #print(transfer_and_print(ix2word_train,ix2word_fix,input_.numpy(),2))
	            #print(transfer_and_print(ix2word_train,ix2word_fix,target.numpy(),1))
	            #i +=1
            #print('input_:\n' + str(input_.numpy()))
            #print('target:\n' + str(target.numpy()))
            #print(input_.size())
            #print(data_.size())
            #print('data_:\n' + str(data_))
            output, _ = model(input_.cuda())
            loss = criterion(output, target.view(-1))
            #print("loss: " + str(loss))
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
            correct += pred.eq(target_test).cpu().sum()  # 对预测正确的数据个数进行累加
            total += target_test.size()[0]
            #correct += find_in_ten(output_test.data,target_test.data)

        test_loss /= iii
        print(epoch)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss , correct, total,
         100. * correct / total))

        testinput, testtarget = data_[1:3, :], data_[3, :]
        #print(testinput)
        testinput=np.transpose(testinput.cpu().numpy()).tolist()
        testtarget=np.transpose(testtarget.cpu().numpy()).tolist()
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

    #hello_idx = torch.LongTensor([word2ix_fix["最后更新时间"]])
    #hello_idx = Variable(hello_idx)
    #hello_embed = model.embeddingsfix(hello_idx)
    #print("after train: \n")
    #print(hello_embed)

    t.save(model.state_dict(), '%s_%s.pth' % ("testforgru", epoch))


def final_test(testresult,testtarget):
    #testresult 2 demension[testsetsize,10],testtarget [testsetsize] 
    count = 0
    rankingnum = 0
    for i, predict in enumerate(testresult):
        if testtarget[i] in predict:
            count += 1       
        try:
            result = predict.index(testtarget[i])
        except:
            result = 10
        rankingnum += result
    return count / len(testtarget) , rankingnum / len(testtarget)

def generate_test(model, start_words, ix2word_train, word2ix_train, ix2word_fix, word2ix_fix):
    # the input is the no. of the word [d,K]
    final_result = []
    for start_word in start_words:

        results = start_word
        start_word_len = len(start_word)
        # 手动设置第一个词为<START>
        input = t.Tensor([word2ix_train['<START>']]).view(1, 1).long()
        if opt.use_gpu: input = input.cuda()
        hidden = None

        #生成    
        for i in range(start_word_len+1):
            if i == 0:
                #要train的idx
                output, hidden = model(input, hidden, 1 , 0)
                w = results[i]
                #print(w)
                input = input.data.new([w]).view(1, 1)
            elif i == 1:
                #这里的input是部门所以要还是要train_idx
                output, hidden = model(input, hidden, 1, 0)
                w = results[i]
                input = input.data.new([w]).view(1, 1)
            else:
                #这里的前一个就是关键词了所以要用的应该是fix的idx
                output, hidden = model(input, hidden, 1 ,1)
                top_indexes = output.data[0].topk(10)[1]
                #print("testing")
                #print(output.data[0].topk(10))
                #print(ix2word[196])
                #print(results)
                for ii in range(len(top_indexes)):
                    if top_indexes[ii].item() in ix2word_train:
                        results.append(top_indexes[ii].item())
        final_result.append(results)
    return final_result



if __name__ == '__main__':
    import fire

    fire.Fire()

