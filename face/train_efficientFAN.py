import os
import torch
import torch.nn as nn
import ref2 as ref # cofw 쓸 때는 ref2 쓴다.
# import ref
from utils.utils import AverageMeter
from utils.eval import compute_error, compute_error_direct, compute_error_direct_efficientFAN
from progress.bar import Bar
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import time
from math import log
import pdb


from torch import nn, optim
from torch.utils.data import (Dataset,
                               DataLoader,
                               TensorDataset)
import tqdm


def weighted_mse_loss(prediction, target, weight=1.0):
    return torch.sum(weight*(prediction-target)**2)/prediction.shape[0]

def attention_distill(prediction, target, weight = 1.0):
    '''
    prediction : student, tensor
    [nb, 640,8,8] , [nb, 256,16,16], [nb, 256,32,32], [nb, 256,64,64]
    target : teacher, tensor
    [nb, 640,8,8] , [nb, 256,16,16], [nb, 256,32,32], [nb, 256,64,64]
    '''
    return torch.sum(weight*(prediction-target)**2)/prediction.shape[0]


def feature_aligned_distill(prediction, target, weight=1e-5):
    return torch.sum(weight*(prediction-target)**2)/prediction.shape[0]

def patch_similarity_distill(prediction, target, weight=1e-5):
    '''
    prediction : list, 각 원소의 크기는 (nb, 4**(n-k), 4**(n-k))
    prediction[k] : tensor, patch size 2**k 일때의 affinity graph matrix. shape : (nb, 4**(n-k), 4**(n-k))
    target : list
    '''
    nb = prediction[0].shape[0]
    n = len(prediction)

    loss = 0.0
    for k in range(n):
        assert(prediction[k].shape[1] == 4**(n-k))
        loss += torch.sum((prediction[k] - target[k]) ** 2) / ((16 ** (n - k)) * n * nb)

    # print(f'patch_sim : n : {n}, loss : {loss * weight}')
    return loss * weight

def attention_featuremap_loss(prediction, target, weight = 1.0):
    '''
    gcnet : weight=0.005, opt.weight2 = 45.43
    4차원 텐서에서만 적용 가능한 loss function
    '''
    nb = prediction.shape[0]
    nc = prediction.shape[1]
    H = prediction.shape[2]
    W = prediction.shape[3]

    pred_norm = torch.norm(prediction,dim=(2,3),keepdim=True) + 1e-10 # [N,C,1,1]
    assert pred_norm.shape == (nb, nc, 1, 1)
    # 각 채널마다 정규화 시켜준다. 즉, i번 채널의 feature map은 해당 feature map의 norm으로 정규화 된다.
    prediction = prediction / pred_norm
    assert prediction.shape == (nb, nc, H,W)

    tar_norm = torch.norm(target, dim=(2, 3),keepdim=True) + 1e-10  # [N,C,1,1]
    assert tar_norm.shape == (nb, nc, 1, 1)
    # 각 채널마다 정규화 시켜준다. 즉, i번 채널의 feature map은 해당 feature map의 norm으로 정규화 된다.
    target = target / tar_norm
    assert target.shape == (nb, nc, H, W)

    return weighted_mse_loss(prediction,target,weight)

def weighted_l1_loss(prediction, target, weight=1.0):
    return torch.sum(weight*torch.abs(prediction-target))/prediction.shape[0]

class WingLoss(nn.Module):
    def __init__(self, w=15, curvature=3):
        super().__init__()
        self.w, self.curvature = w, curvature
        self.C = w - w * log((1 + w / curvature))

    def forward(self, y_hat, y):
        #print(f'y_hat : {y_hat.shape}, y.view() : {y.view(y.shape[0], -1).shape}')

        # diff = y_hat.view(y_hat.shape[0],-1) - y.view(y.shape[0], -1) #
        diff = y_hat - y
        diff_abs = diff.abs()
        loss = diff_abs.clone() # torch.Size([nb, 98, 2])

        idx_smaller = diff_abs < self.w
        idx_bigger = diff_abs >= self.w

        loss[idx_smaller] = self.w * torch.log(1 + loss[idx_smaller].abs() / self.curvature)
        loss[idx_bigger] = loss[idx_bigger].abs() - self.C
        loss = loss.mean()
        return loss

def train_kd_att_2(epoch, opt, train_loader, model, teacher, attn_loss, optimizer):
    '''
    해당 train함수는 1 epoch만 도는것임.
    main에서 epoch만큼 train함수를 호출하기 때문에 총 epoch만큼 실행하게 된다.

    - We set 𝜔 = 15 and 𝜀 = 3 for Wing loss [5], to calculate the 𝐿_P and the weight factor 𝜆 = e−5.
    ==> Wing loss, L_P가 뭔지 알아야함.
    '''

    split = 'train'
    model.train()
    attn_loss.train()
    teacher.eval()

    # Wing loss test code
    wingloss = WingLoss(w=15,curvature=3)
    # wingloss_kd = [WingLoss(w=2,curvature=0.5), WingLoss(w=4,curvature=1), WingLoss(w=8,curvature=2), WingLoss(w=8,curvature=2)]

    # Initialize evaluations
    cost = AverageMeter() # cost는 loss이고
    kd_cost = AverageMeter()
    error2d = [] # error는 model의 성능을 나타내기 위한 지표. like NME, 정확도, mAP, ...
    for i in range(2): error2d.append(AverageMeter())

    num_iters = len(train_loader)
    bar = Bar('==>', max=num_iters)
    for i, (img, hmap, pts, norm_factor) in enumerate(train_loader):
        img = img.to("cuda")
        hmap = hmap.to("cuda")
        pts = pts.to("cuda")
        norm_factor = norm_factor.to("cuda")
        # print(f'norm_factor shape: {norm_factor.shape}, norm type : {type(norm_factor)}')

        # Constants
        num_batch = img.shape[0] # batch size
        num_joint = ref.num_landmarks # landmark number
        # Forward propagation
        outputs = model(img)

        # teacher로부터 나온 output에서는 gradient를 계산하지 않는다.
        outputs_t = teacher(img)
        for i in range(len(outputs_t)):
            # outputs_t[2,3,4,5] : tensor type
            outputs_t[i] = Variable(outputs_t[i], requires_grad=False)

        # Compute loss
        # 바꿔야하는 부분. mse_loss ==> wing loss
        loss = 0.0
        loss += weighted_mse_loss(outputs[0], hmap)
        loss += wingloss(outputs[1], pts)*opt.weight1 # wing loss test code

        # Compute attention kd loss
        for i in range(2,5+1):
            # assert (outputs[i].shape == outputs_t[i].shape)
            att_loss = attn_loss(outputs[2], outputs[3],outputs[4],outputs[5],
                                  outputs_t[2],outputs_t[3],outputs_t[4],outputs_t[5])

        att_loss = att_loss * opt.weight2
        loss = loss + att_loss

        # if loss.isnan().any() == True:
        #     import pdb
        #     pdb.set_trace()
        # Update model parameters with backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # debugging code
        # outputs = model(img)
        # if outputs[2].isnan().any() == True:
        #     import pdb
        #     pdb.set_trace()
        # end debugging

        # Update evaluations
        cost.update((loss-att_loss).detach().item(), num_batch) # detach()를 해야 tensor가 미분을 위해 추적하는 것을 막을 수 있다.
        kd_cost.update(att_loss.detach().item(), num_batch)

        error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([num_batch, num_joint]).to("cuda")))
        error2d[1].update(compute_error_direct_efficientFAN((outputs[1]).detach(), pts.detach(), norm_factor, torch.ones([num_batch, num_joint]).to("cuda"))) #NME

        # msg : epoch / 1 epoch 안에서의 iter / 1개 epoch 안에서의 총 iter /  train,val / total : 1 epoch하면서 지금까지 걸린 시간 / eta : 예상 남은 시간 / cost / lr
        msg = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)
        if split == 'train':
            msg = '{} | LR: {:1.1e}'.format(msg, optimizer.param_groups[0]['lr'])

        msg = '{} | GT_Cost {cost.avg:.2f}'.format(msg, cost=cost)
        msg = '{} | KD_Cost {cost.avg:.2f}'.format(msg, cost=kd_cost)
        for j in range(2):
            msg = '{} | E{:d} {:.2f}'.format(msg, j, error2d[j].avg)
        Bar.suffix = msg
        bar.next()

    bar.finish()

    return [cost.avg, kd_cost.avg], [error2d[0].avg, error2d[1].avg]

def train_kd_att(epoch, opt, train_loader, model, teacher, attention_student,attention_teacher,optimizer):
    '''
    해당 train함수는 1 epoch만 도는것임.
    main에서 epoch만큼 train함수를 호출하기 때문에 총 epoch만큼 실행하게 된다.

    - We set 𝜔 = 15 and 𝜀 = 3 for Wing loss [5], to calculate the 𝐿_P and the weight factor 𝜆 = e−5.
    ==> Wing loss, L_P가 뭔지 알아야함.
    '''

    split = 'train'
    model.train()
    attention_teacher.train()
    attention_student.train()
    teacher.eval()

    # Wing loss test code
    wingloss = WingLoss(w=15,curvature=3)
    # wingloss_kd = [WingLoss(w=2,curvature=0.5), WingLoss(w=4,curvature=1), WingLoss(w=8,curvature=2), WingLoss(w=8,curvature=2)]

    # Initialize evaluations
    cost = AverageMeter() # cost는 loss이고
    att_cost = AverageMeter()

    error2d = [] # error는 model의 성능을 나타내기 위한 지표. like NME, 정확도, mAP, ...
    for i in range(2): error2d.append(AverageMeter())

    num_iters = len(train_loader)
    bar = Bar('==>', max=num_iters)
    for i, (img, hmap, pts, norm_factor) in enumerate(train_loader):
        img = img.to("cuda")
        hmap = hmap.to("cuda")
        pts = pts.to("cuda")
        norm_factor = norm_factor.to("cuda")
        # print(f'norm_factor shape: {norm_factor.shape}, norm type : {type(norm_factor)}')

        # Constants
        num_batch = img.shape[0] # batch size
        num_joint = ref.num_landmarks # landmark number
        # Forward propagation
        outputs = model(img)
        outputs_att_student = attention_student(outputs[2],outputs[3],outputs[4],outputs[5]) # list : [B1_, B2_, B3_, B4_]

        # teacher로부터 나온 output에서는 gradient를 계산하지 않는다.
        outputs_t = teacher(img)
        for i in range(len(outputs_t)):
            # outputs_t[2,3,4,5] : tensor type
            outputs_t[i] = Variable(outputs_t[i], requires_grad=False)

        outputs_att_teacher = attention_teacher(outputs_t[2],outputs_t[3],outputs_t[4],outputs_t[5]) # list : [B1_, B2_, B3_, B4_]
        # Compute loss
        # 바꿔야하는 부분. mse_loss ==> wing loss
        loss = 0.0
        loss += weighted_mse_loss(outputs[0], hmap)
        loss += wingloss(outputs[1], pts)*opt.weight1 # wing loss test code

        # Compute attention kd loss
        att_loss = 0.0
        # for i in range(len(outputs_att_student)):
        for i in range(len(outputs_att_student)-2):
            assert(outputs_att_student[i].shape == outputs_att_teacher[i].shape)
            att_loss += attention_featuremap_loss(outputs_att_student[i], outputs_att_teacher[i], weight=0.128533987)

        att_loss = att_loss * opt.weight2
        # print(f' ')
        # print(f'fa_loss : {fa_loss}')
        # print(f'ps_loss : {ps_loss}')
        # print(f'loss : {loss}')
        loss = loss + att_loss

        # Update model parameters with backpropagation
        optimizer.zero_grad() #optimizer 선정도 opt에서 하는건가?
        loss.backward()
        optimizer.step()
        # Update evaluations
        cost.update((loss-att_loss).detach().item(), num_batch) # detach()를 해야 tensor가 미분을 위해 추적하는 것을 막을 수 있다.
        att_cost.update(att_loss.detach().item(), num_batch)

        error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([num_batch, num_joint]).to("cuda")))
        error2d[1].update(compute_error_direct_efficientFAN((outputs[1]).detach(), pts.detach(), norm_factor, torch.ones([num_batch, num_joint]).to("cuda"))) #NME

        # msg : epoch / 1 epoch 안에서의 iter / 1개 epoch 안에서의 총 iter /  train,val / total : 1 epoch하면서 지금까지 걸린 시간 / eta : 예상 남은 시간 / cost / lr
        msg = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)
        if split == 'train':
            msg = '{} | LR: {:1.1e}'.format(msg, optimizer.param_groups[0]['lr'])

        msg = '{} | GT_Cost {cost.avg:.2f}'.format(msg, cost=cost)
        msg = '{} | Att_Cost {cost.avg:.2f}'.format(msg, cost=att_cost)

        for j in range(2):
            msg = '{} | E{:d} {:.2f}'.format(msg, j, error2d[j].avg)
        Bar.suffix = msg
        bar.next()

    bar.finish()

    return [cost.avg, att_cost.avg], [error2d[0].avg, error2d[1].avg]

def train_kd(epoch, opt, train_loader, model, teacher, optimizer):
    '''
    해당 train함수는 1 epoch만 도는것임.
    main에서 epoch만큼 train함수를 호출하기 때문에 총 epoch만큼 실행하게 된다.

    - We set 𝜔 = 15 and 𝜀 = 3 for Wing loss [5], to calculate the 𝐿_P and the weight factor 𝜆 = e−5.
    ==> Wing loss, L_P가 뭔지 알아야함.
    '''

    split = 'train'
    model.train()
    teacher.eval()

    # Wing loss test code
    wingloss = WingLoss(w=15,curvature=3)

    # Initialize evaluations
    cost = AverageMeter() # cost는 loss이고
    kd_cost = AverageMeter()
    error2d = [] # error는 model의 성능을 나타내기 위한 지표. like NME, 정확도, mAP, ...
    for i in range(2): error2d.append(AverageMeter())

    num_iters = len(train_loader)
    bar = Bar('==>', max=num_iters)
    for i, (img, hmap, pts, norm_factor) in enumerate(train_loader):
        img = img.to("cuda")
        hmap = hmap.to("cuda")
        pts = pts.to("cuda")
        norm_factor = norm_factor.to("cuda")
        # print(f'norm_factor shape: {norm_factor.shape}, norm type : {type(norm_factor)}')

        # Constants
        num_batch = img.shape[0] # batch size
        num_joint = ref.num_landmarks # landmark number
        # Forward propagation
        outputs = model(img)

        # teacher로부터 나온 output에서는 gradient를 계산하지 않는다.
        outputs_t = teacher(img)
        for i in range(len(outputs_t)):
            # outputs_t[2,3,4,5] : tensor type
            outputs_t[i] = Variable(outputs_t[i], requires_grad=False)

        # Compute loss
        # 바꿔야하는 부분. mse_loss ==> wing loss
        loss = 0.0
        loss += weighted_mse_loss(outputs[0], hmap)
        loss += wingloss(outputs[1], pts)*opt.weight1 # wing loss test code

        fa_loss = 0.0
        # ps_loss = 0.0
        # feature alignment Loss
        for i in range(2, 5+1):
            assert (outputs[i].shape == outputs_t[i].shape)
            fa_loss += feature_aligned_distill(outputs[i],outputs_t[i], weight=5e-4) # 5e-4 일때 gt loss : kd loss = 2:1

        fa_loss = fa_loss*opt.weight2
        # Patch Similarity Loss
        '''for i in range(2, 5+1):
            assert (len(outputs[i]) == len(outputs_t[i]))
            ps_loss += patch_similarity_distill(outputs[i],outputs_t[i], weight=200.0)'''

        # print(f' ')
        # print(f'fa_loss : {fa_loss}')
        # print(f'ps_loss : {ps_loss}')
        # print(f'loss : {loss}')
        loss = loss + fa_loss

        # Update model parameters with backpropagation
        optimizer.zero_grad() #optimizer 선정도 opt에서 하는건가?
        loss.backward()
        optimizer.step()
        # Update evaluations
        cost.update((loss-fa_loss).detach().item(), num_batch) # detach()를 해야 tensor가 미분을 위해 추적하는 것을 막을 수 있다.
        kd_cost.update(fa_loss.detach().item(), num_batch)

        error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([num_batch, num_joint]).to("cuda")))
        error2d[1].update(compute_error_direct_efficientFAN((outputs[1]).detach(), pts.detach(), norm_factor, torch.ones([num_batch, num_joint]).to("cuda"))) #NME

        # msg : epoch / 1 epoch 안에서의 iter / 1개 epoch 안에서의 총 iter /  train,val / total : 1 epoch하면서 지금까지 걸린 시간 / eta : 예상 남은 시간 / cost / lr
        msg = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)
        if split == 'train':
            msg = '{} | LR: {:1.1e}'.format(msg, optimizer.param_groups[0]['lr'])

        msg = '{} | GT_Cost {cost.avg:.2f}'.format(msg, cost=cost)
        msg = '{} | KD_Cost {cost.avg:.2f}'.format(msg, cost=kd_cost)
        for j in range(2):
            msg = '{} | E{:d} {:.2f}'.format(msg, j, error2d[j].avg)
        Bar.suffix = msg
        bar.next()

    bar.finish()

    return [cost.avg, kd_cost.avg], [error2d[0].avg, error2d[1].avg]

# 여기서 바꿀거라고는 opt클래스 내용, Wing loss직접 적는것밖에 없는듯.
# 그리고 1번 epoch돌때마다 test loss 판단하는 프로세스로 수정하자. 이건 main에서 해야하나??
def train(epoch, opt, train_loader, model, optimizer):
    '''
    해당 train함수는 1 epoch만 도는것임.
    main에서 epoch만큼 train함수를 호출하기 때문에 총 epoch만큼 실행하게 된다.

    - We set 𝜔 = 15 and 𝜀 = 3 for Wing loss [5], to calculate the 𝐿_P and the weight factor 𝜆 = e−5.
    ==> Wing loss, L_P가 뭔지 알아야함.
    '''

    split = 'train'
    model.train()

    # Wing loss test code
    wingloss = WingLoss(w=15,curvature=3)

    # Initialize evaluations
    cost = AverageMeter() # cost는 loss이고
    error2d = [] # error는 model의 성능을 나타내기 위한 지표. like NME, 정확도, mAP, ...
    for i in range(2): error2d.append(AverageMeter())

    num_iters = len(train_loader)
    bar = Bar('==>', max=num_iters)
    for i, (img, hmap, pts, norm_factor) in enumerate(train_loader):
        img = img.to("cuda")
        hmap = hmap.to("cuda")
        pts = pts.to("cuda")
        norm_factor = norm_factor.to("cuda")
        # print(f'norm_factor shape: {norm_factor.shape}, norm type : {type(norm_factor)}')

        # Constants
        num_batch = img.shape[0] # batch size
        num_joint = ref.num_landmarks # landmark number
        # Forward propagation
        outputs = model(img)
        # Compute cost
        loss = 0.0
        # Compute loss
        # 바꿔야하는 부분. mse_loss ==> wing loss
        if opt.integral == 0:
            loss += weighted_mse_loss(outputs[0], hmap)
        elif opt.integral == 1:
            loss += weighted_mse_loss(outputs[0], hmap)
            loss += wingloss(outputs[1], pts)*opt.weight1 # wing loss test code

        # Update model parameters with backpropagation
        optimizer.zero_grad() #optimizer 선정도 opt에서 하는건가?
        loss.backward()
        optimizer.step()
        # Update evaluations
        cost.update(loss.detach().item(), num_batch) # detach()를 해야 tensor가 미분을 위해 추적하는 것을 막을 수 있다.

        error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([num_batch, num_joint]).to("cuda")))
        error2d[1].update(compute_error_direct_efficientFAN((outputs[1]).detach(), pts.detach(), norm_factor, torch.ones([num_batch, num_joint]).to("cuda"))) #NME
        # error2d[1].update(compute_error_direct((outputs[1]).detach(), pts.detach(), torch.ones([num_batch, num_joint]).to("cuda"))) #NME

        # msg : epoch / 1 epoch 안에서의 iter / 1개 epoch 안에서의 총 iter /  train,val / total : 1 epoch하면서 지금까지 걸린 시간 / eta : 예상 남은 시간 / cost / lr
        msg = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)
        if split == 'train':
            msg = '{} | LR: {:1.1e}'.format(msg, optimizer.param_groups[0]['lr'])
        msg = '{} | Cost {cost.avg:.2f}'.format(msg, cost=cost)
        for j in range(2):
            msg = '{} | E{:d} {:.2f}'.format(msg, j, error2d[j].avg)
        Bar.suffix = msg
        bar.next()

    bar.finish()

    return cost.avg, [error2d[0].avg, error2d[1].avg]

def val(epoch, opt, val_loader, model):
    split = 'val'
    model.eval()

    # Initialize evaluations
    wingloss = WingLoss(w=15,curvature=3)
    cost = AverageMeter()
    error2d = []
    for i in range(2): error2d.append(AverageMeter())
    
    num_iters = len(val_loader)
    bar = Bar('==>', max=num_iters)

    for i, (img, hmap, pts, norm_factor) in enumerate(val_loader):
        img = img.to("cuda")
        hmap = hmap.to("cuda")
        pts = pts.to("cuda")
        norm_factor = norm_factor.to("cuda")

        # Constants
        nb = img.shape[0]
        nj = ref.num_landmarks

        # Forward propagation
        outputs = model(img)

        # Compute cost
        loss = 0.0

        if opt.integral == 0:
            loss += weighted_mse_loss(outputs[0], hmap)
        elif opt.integral == 1:
            loss += weighted_mse_loss(outputs[0], hmap)
            loss += wingloss(outputs[1], pts)*opt.weight1

        # Update evaluations
        cost.update(loss.detach().item(), nb)
        if opt.network == 'fan' or opt.network == 'fan-trained':
            error2d[0].update(compute_error(outputs[opt.num_modules-1].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")))
        else:
            if opt.integral == 0:
                error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")))
            elif opt.integral == 1:
                error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")))
                error2d[1].update(compute_error_direct_efficientFAN((outputs[1]).detach(), pts.detach(), norm_factor, torch.ones([nb, nj]).to("cuda")))

        msg = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)

        msg = '{} | Cost {cost.avg:.2f}'.format(msg, cost=cost)
        for j in range(2):
            msg = '{} | E{:d} {:.2f}'.format(msg, j, error2d[j].avg)
        Bar.suffix = msg
        bar.next()

    bar.finish()

    return cost.avg, [error2d[0].avg, error2d[1].avg]


def test(epoch, opt, test_loader, model, name):
    split = 'test'
    model.eval()

    # Initialize evaluations
    wingloss = WingLoss(w=15,curvature=3)
    cost = AverageMeter()
    error2d = []
    for i in range(2): error2d.append(AverageMeter())
    nme_list = []
    time_all = 0.0

    num_iters = len(test_loader)
    bar = Bar('==>', max=num_iters)
    
    # Save results
    if opt.save_results == True:
        heatmaps = []
        landmarks = []

    for i, (img, hmap, pts, norm_factor) in enumerate(test_loader):
        img = img.to("cuda")
        hmap = hmap.to("cuda")
        pts = pts.to("cuda")
        norm_factor = norm_factor.to("cuda")

        # Constants
        nb = img.shape[0]
        nj = ref.num_landmarks

        # Forward propagation
        time1 = time.time()
        outputs = model(img)
        time2 = time.time()
        time_all += (time2-time1)

        # Compute cost
        loss = 0.0
        if opt.network == 'fan' or opt.network == 'fan-trained':
            for j in range(opt.num_modules):
                loss += weighted_mse_loss(outputs[j], hmap)
        else:
            if opt.integral == 0:
                loss += weighted_mse_loss(outputs[0], hmap)
            elif opt.integral == 1:
                loss += weighted_mse_loss(outputs[0], hmap)
                loss += wingloss(outputs[1], pts)*opt.weight1

        # Update evaluations
        cost.update(loss.detach().item(), nb)
        if opt.network == 'fan' or opt.network == 'fan-trained':
            error2d[0].update(compute_error(outputs[opt.num_modules-1].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")))
        else:
            if opt.integral == 0:
                error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")))
            elif opt.integral == 1:
                error2d[0].update(compute_error(outputs[0].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")))
                error2d[1].update(compute_error_direct_efficientFAN((outputs[1]).detach(), pts.detach(), norm_factor, torch.ones([nb, nj]).to("cuda")))

        # Update NME
        if opt.network == 'fan' or opt.network == 'fan-trained':
            nme_list.append(compute_error(outputs[opt.num_modules-1].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")).item())
        else:
            if opt.integral == 0:
                nme_list.append(compute_error(outputs[0].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")).item())
            elif opt.integral == 1:
                nme_list.append(compute_error_direct(outputs[1].detach(), pts.detach(), torch.ones([nb, nj]).to("cuda")).item())

        # Save results
        if opt.save_results == True:
            heatmaps.append(outputs[0].detach().to("cpu"))
            landmarks.append(outputs[1].detach().to("cpu"))

        msg = '{split} Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:}'.format(epoch, i, num_iters, split=split, total=bar.elapsed_td, eta=bar.eta_td)

        msg = '{} | Cost {cost.avg:.2f}'.format(msg, cost=cost)
        for j in range(2):
            msg = '{} | E{:d} {:.2f}'.format(msg, j, error2d[j].avg)
        Bar.suffix = msg
        bar.next()

    bar.finish()

    # Final evaluation
    nme_sorted = np.sort(nme_list)
    num_samples = nme_sorted.shape[0]
    bin = np.arange(0, 7.01, 0.05) #??
    num_bin = bin.shape[0]
    pdf = np.zeros(num_bin)
    for i in range(num_samples):
        val = nme_sorted[i]
        idx = min(int(val/0.05)+1, num_bin-1)
        pdf[idx] += 1
    cdf = np.cumsum(pdf)
    cdf = cdf / num_samples * 100.0
    resultname = os.path.join(opt.save_dir, 'result_%s_cdf.txt' % name)
    np.savetxt(resultname, cdf)

    # Compute AUC
    AUC = 0.0
    for i in range(num_bin-1):
        AUC += cdf[i+1]
    AUC /= (num_bin-1)
    print('AUC for 7%% = %.2f' % AUC)
    resultname = os.path.join(opt.save_dir, 'result_%s_auc.txt' % name)
    file = open(resultname, 'w')
    file.write('AUC for 7%% = %.2f' % AUC)
    file.close()

    # Compute time
    time_average = time_all / num_iters
    fps = 1.0 / time_average
    print('time for one frame = %.2f (s)' % time_average)
    print('fps = %.2f (frame/s)' % fps)
    resultname = os.path.join(opt.save_dir, 'result_%s_time.txt' % name)
    file = open(resultname, 'w')
    file.write('time for one frame = %.2f (ms)\n' % (time_average*1000.0))
    file.write('fps = %.2f (frame/s)' % fps)
    file.close()

    # Save results
    if opt.save_results == True:
        heatmaps = torch.cat(heatmaps, 0)
        filename = os.path.join(opt.save_dir, 'heatmaps_%s.pth' % name)
        torch.save(heatmaps, filename)
        landmarks = torch.cat(landmarks, 0)
        filename = os.path.join(opt.save_dir, 'landmarks_%s.pth' % name)
        torch.save(landmarks, filename)

    if False:
        # Constants for figure
        fontsize = 12
        figure_size = (11, 6)

        # Figure
        fig = plt.figure()
        ax = plt.gca()
        plt.plot(bin, cdf, linestyle='-', linewidth=2)
        ax.set_xlabel('NME (%)')
        ax.set_ylabel('Test Images (%)')
        ax.set_title('300W-Testset, NME (%)', fontsize=fontsize)
        ax.set_xlim([0., 7.])
        ax.set_ylim([0., 100.])
        ax.set_yticks(np.arange(0., 101., 10.))
        plt.grid('on', linestyle='--', linewidth=0.5)
        #fig.set_size_inches(np.asarray(figure_size))
        plt.show()

        resultname = os.path.join(opt.save_dir, 'result_cdf.jpg')
        fig.savefig(resultname)

    return cost.avg, [error2d[0].avg, error2d[1].avg]


def final_plot(model,val_loader):
    import cv2

    split = 'val'
    model.eval()

    for i, (img, hmap, pts, norm_factor) in enumerate(val_loader):
        img = img.to("cuda")
        hmap = hmap.to("cuda")
        pts = pts.to("cuda")
        norm_factor = norm_factor.to("cuda")

        # Constants
        nb = img.shape[0]
        nj = ref.num_landmarks

        # Forward propagation
        outputs = model(img)

        pts = pts.detach().cpu().numpy()
        img = img.detach().cpu()

        # import pdb
        # pdb.set_trace()

        cnt = 1
        for i in range(nb):
            x = []
            y = []

            for j in range(nj):
                x.append(pts[i][j][0])
                y.append(pts[i][j][1])

            image = img[i].permute(1, 2, 0)
            image = image.numpy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image) # 사진 파랗게 나옴.
            plt.scatter(x,y, c='y')


            plt.subplot(2,5,cnt)

            plt.show()

        # import pdb
        # pdb.set_trace()
        # exit(0)

