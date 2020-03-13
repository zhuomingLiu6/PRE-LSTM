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
from torchnet import meter

#filepath = r'C:\Users\XPS\Desktop\sequence.txt'
#tablepath = r'C:\Users\XPS\Desktop\唯一表格名字.txt'

class TrainingModel(nn.Module):
    #vocab_size mean how many distinct word would appear in the poem
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(TrainingModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2)#lstm里面有参数可以通过梯度下降惊醒拟合，通过input得到输入和输出然后计算loss返回提反独下降
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        #print('test')
        #print(input.size())
        seq_len, batch_size = input.size()
        #print(seq_len)
        #print(batch_size)
        if hidden is None:
            #  h_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #  c_0 = 0.01*torch.Tensor(2, batch_size, self.hidden_dim).normal_().cuda()
            #print('test')
            #print(type(input))
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        # size: (seq_len,batch_size,embeding_dim)
        embeds = self.embeddings(input)
        # output size: (seq_len,batch_size,hidden_dim)
        output, hidden = self.lstm(embeds, (h_0, c_0))

        # size: (seq_len*batch_size,vocab_size)
        #print("beforeenter")
        #print(output.view(seq_len * batch_size, -1).size())
        output = self.linear1(output.view(seq_len * batch_size, -1))
        #print("after_enter")
        #print(output.size())
        return output, hidden

class Config(object):
	use_gpu = True
	#env = 'mytest'  # visdom env
	parsed_data_path = r'/home/liuzhm/gpu_version/sequence_without_vec.npz'
	batch_size = 128
	model_path = r"/data1/pcr/unclebao/sequence.npz" # 预训练模型路径
	epoch = 20
	plot_every = 10  # 每20个batch 可视化一次
	debug_file = '/tmp/debugp'
	lr = 1e-3
	start_words = ['法院','工商登记号']  # 诗歌开始

opt = Config()

def generate_parsed_data():
	file_object=open(filepath,'r',encoding='utf-8')
	chunk_data = file_object.read()
	data = []
	chunk_data = chunk_data.split('\n')
	for seq in chunk_data:
		data.append(seq.split(','))

	testset = {item for row in data for item in row}
	word2ix = {_word: _ix for _ix, _word in enumerate(testset)}
	word2ix['<EOP>'] = len(word2ix)  # 终止标识符
	word2ix['<START>'] = len(word2ix)  # 起始标识符
	word2ix['</s>'] = len(word2ix)  # 空格
	ix2word = {_ix: _word for _word, _ix in list(word2ix.items())}

	for i in range(len(data)):
		data[i] = ["<START>"] + list(data[i]) + ["<EOP>"]

	new_data = [[word2ix[_word] for _word in _sentence] for _sentence in data]

	np.savez_compressed("sequence.npz",
	                        data=new_data,
	                        word2ix=word2ix,
	                        ix2word=ix2word)
	return new_data, word2ix, ix2word

def load_data(parsed_data_path):
	data = np.load(parsed_data_path)
	data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
	return data, word2ix, ix2word

def transfer_and_print(ix2word,data):
	final = []
	for row in data:
		temp = []
		for item in row:
			temp.append(ix2word[item])
		final.append(temp)
	return final



def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v.strip("'"))
    opt.device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    device = opt.device
    #vis = Visualizer(env=opt.env)
    loss_meter_inornot = meter.AverageValueMeter()
    loss_meter_rankingnum = meter.AverageValueMeter()

    # 获取数据
    data, word2ix, ix2word = load_data(opt.parsed_data_path)


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

    loss_meter = meter.AverageValueMeter()

    #预训练模型
    '''
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))'''
    model.to(device)
    
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
	            print(transfer_and_print(ix2word,input_.numpy()))
	            print(transfer_and_print(ix2word,target.numpy()))
	            i +=1'''
            #print('input_:\n' + str(input_.numpy()))
            #print('target:\n' + str(target.numpy()))
            #print(input_.size())
            #print(data_.size())
            #print('data_:\n' + str(data_))
            output, _ = model(input_.cuda())
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

        print("loss_meter.value()[0]: " + str(loss_meter.value()[0]))
        print("loss.item() :" + str(loss.item()))

        #测试集上测试
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

        testinput, testtarget = data_[1:3, :], data_[3, :]
        #print(testinput)
        testinput=np.transpose(testinput.cpu().numpy()).tolist()
        testtarget=np.transpose(testtarget.cpu().numpy()).tolist()
        #print(testinput)
        #print(testtarget)
        testresult = generate(model, testinput, ix2word, word2ix)
        #print(testresult)
        inornot,rankingnum = final_test(testresult,testtarget)
        loss_meter_inornot.add(inornot)
        loss_meter_rankingnum.add(rankingnum)
        print("inornot: " + str(loss_meter_inornot.value()[0]))
        print("rankingnum: " + str(loss_meter_rankingnum.value()[0]))     

        #vis.plot('lossintrain',  test_loss)
        #vis.plot('lossintrainrate', correct / len(dataloader_fortest.dataset))
        model.train() 
    #hello_idx = torch.LongTensor([word2ix["实收资本"]])
    #hello_idx = Variable(hello_idx)
    #hello_embed = model.embeddings(hello_idx)
    #print("after train: \n")
    #print(hello_embed)

    t.save(model.state_dict(), '%s_%s.pth' % ("test", epoch))


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

def generate(model, start_words, ix2word, word2ix):
    final_result = []
    #print(start_words)
    for start_word in start_words:
        #print("test")
        results = start_word
        start_word_len = len(start_word)
        # 手动设置第一个词为<START>
        input = t.Tensor([word2ix['<START>']]).view(1, 1).long()
        if opt.use_gpu: input = input.cuda()
        hidden = None
        #print(results)
        #生成
        for i in range(start_word_len+1):
            output, hidden = model(input, hidden)

            if i < start_word_len:
                w = results[i]
                input = input.data.new([w]).view(1, 1)
            else:
                top_indexes = output.data[0].topk(10)[1]
                #print("testing")
                #print(output.data[0].topk(10))
                #print(ix2word[196])
                for i in range(len(top_indexes)):
                    results.append(top_indexes[i].item())
        final_result.append(results)
    return final_result




if __name__ == '__main__':
    import fire

    fire.Fire()

'''
#测试词向量
data, word2ix, ix2word = load_data(opt.parsed_data_path)
data = t.from_numpy(data)
dataloader = t.utils.data.DataLoader(data,
                                     batch_size=opt.batch_size,
                                     shuffle=True,
                                     num_workers=1)


model = TrainingModel(len(word2ix), 128, 256)
optimizer = t.optim.Adam(model.parameters(), lr=opt.lr)
criterion = nn.CrossEntropyLoss()
#测试对应的词向量
model.load_state_dict(t.load("testmodel_19.pth"))
#print(model.embeddings.weight.detach().numpy()[1183])

#print("\n")
#print([word2ix['实收资本']])

hello_idx = torch.LongTensor([word2ix['流动人口登记表']])
hello_idx = Variable(hello_idx)
hello_embed = model.embeddings(hello_idx)
print(hello_embed)
#print(model.embeddings.weight.detach().numpy()[0])
#print(model.embeddings.weight.detach().numpy()[1])
'''
'''
#向量推荐
deparment_idx = torch.LongTensor([word2ix['人才办']])
deparment_idx = Variable(deparment_idx)
deparment_embed = model.embeddings(deparment_idx)


key_idx = torch.LongTensor([word2ix['姓名']])
key_idx = Variable(key_idx)
key_embed = model.embeddings(key_idx)

#print(deparment_embed)
#print(key_embed)
#print(deparment_embed+key_embed)

test = deparment_embed+key_embed
#print(t.mm(deparment_embed,t.transpose(key_embed,1,0)))

file_object=open(tablepath,'r',encoding='utf-8')
chunk_data = file_object.read()
file_object.close()
chunk_data = chunk_data.split('\n')

table_idx = []
table_embeds = []
for table in chunk_data:
	idx = Variable(torch.LongTensor([word2ix[table]]))
	table_idx.append(idx)
	table_embeds.append(model.embeddings(idx))

result = []
for table_embed in table_embeds:
	result.append(t.mm(test,t.transpose(table_embed,1,0)))

def takeSecond(elem):
    return elem[1]

tuple_result = []
for i in range(len(result)):
	#print(chunk_data[i])
	#print(result[i].detach().numpy()[0][0])
	tuple_result.append((chunk_data[i],result[i].detach().numpy()[0][0]))
tuple_result.sort(key=takeSecond)
for i in tuple_result:
	print(i[0])
	print(i[1])'''

'''
#生成处理好的数据
generate_parsed_data()'''