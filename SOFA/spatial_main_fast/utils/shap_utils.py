import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim



def gen_concept_masks(gen_model,target_img):
    return gen_model.generate(target_img)

def feat_prob(model,feat,target_label):
    
    with torch.no_grad():
        if model == None:
            probabilities = torch.nn.functional.softmax(feat,dim=1)
        else:
            output = model(feat)
            probabilities = torch.nn.functional.softmax(output,dim=1)
        return probabilities[:,target_label]
    
class NeuralNet(nn.Module): ### a simple NN network
    def __init__(self, in_size,bs,head,lr,ftdim):
        super(NeuralNet, self).__init__()
        torch.manual_seed(0)
        self.model = nn.Sequential(nn.Linear(in_size,ftdim))
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        self.loss = torch.nn.BCELoss()
        self.bs = bs
        self.fc = head

    def change_lr(self,lr):
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)

    def forward(self,x):
        if self.fc != None:
            # print("FC Version")
            return self.fc(self.model(x))
        else:
            return self.model(x)
    
    def forward_feat(self,x):
        return self.model(x)
    
    def step(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        output.backward()
        self.optimizer.step()
        # print(output.detach().cpu().numpy())
        return output.detach().cpu().numpy()
    
    def step_val(self, x,y):
        self.optimizer.zero_grad()
        loss = self.loss
        output = loss(x, y)
        return output.detach().cpu().numpy()


def learning_feat(target_model,full_model,concept_mask,target_img,target_label,fc,image_norm=None):

    target_img = target_img.unsqueeze(0)  
    concept_mask = concept_mask.unsqueeze(1)  

    batch_img = target_img * concept_mask  

    tmp_dl = DataLoader(dataset = batch_img, batch_size = 1000, shuffle =False)
    
    output = None
    fc_res = None
    for x in tmp_dl:
        with torch.no_grad():
            if output == None:
                # print(target_model)
                # print("x.shape",x.shape)
                output = target_model(x.cuda())
                # print(output)
                output = output['avgpool'].squeeze().squeeze()
                fc_res = torch.nn.functional.softmax(fc(output), dim=1)[:,target_label]
            else:
                tmp_out = target_model(x.cuda())
                tmp_out = tmp_out['avgpool'].squeeze().squeeze()
                output = torch.cat((output,tmp_out))
                fc_res = torch.cat((fc_res,torch.nn.functional.softmax(fc(tmp_out), dim=1)[:,target_label]))
        
    return output, fc_res

    
def learn_PIE(target_model,full_model,masks_tmp,target_img,target_label,fc,lr,epochs,image_norm=None):
    flag = 0
    for i in range(10):
    
        num_feat = masks_tmp.shape[0]
        bin_x = torch.bernoulli(torch.full((10000, num_feat), 0.5)).bool()
        new_mask = torch.stack([masks_tmp[i].sum(0) for i in bin_x]).bool()
        
        # print("bin_x",num_feat,bin_x.shape,new_mask.shape)
        
        feat, probs = learning_feat(target_model,full_model,new_mask,target_img,target_label,fc,image_norm)
        feat = feat.detach().clone()#.cpu()
        probs = probs.detach().clone()#.cpu()
        bin_x_torch = torch.tensor(bin_x.tolist(),dtype=torch.float)
        data = [[x,y] for x,y in zip(bin_x_torch,probs)]
        bs = 100
        losses = []
        
        net = NeuralNet(num_feat,bs,fc,lr,feat.shape[1]).cuda()
        
        net.change_lr(lr)
        data_comb_train = DataLoader(dataset = data[num_feat:], batch_size = bs, shuffle =True)
        
        ##### learning combin
        epoch_num = epochs
        for epoch in range(epoch_num):
            loss = 0
            for x,y in data_comb_train:
                # print(x.shape)
                pred = torch.nn.functional.softmax(net(x.cuda()), dim=1)[:,target_label]
                tmploss = net.step(pred,y.cuda())*x.shape[0]
                loss += tmploss
            if epoch==30:
                lr = 0.5*lr
                net.change_lr(lr)
            elif epoch==50:
                lr = 0.5*lr
                net.change_lr(lr)
        if tmploss<100:
            if flag:
                flag = 0
                print("After Warning&Retraining:",i, tmploss)
            break
        else:
            flag = 1
            print("Warning!!! tmploss>=1000. Retraining!!!", i, tmploss)
    net.eval()
    return net
