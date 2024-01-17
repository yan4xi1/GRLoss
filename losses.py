from typing_extensions import final
import torch
from torch._C import ThroughputBenchmark
import torch.nn.functional as F
import math 


def GR_loss(preds,label_vec,K,V,Q,epoch,P=None):
    q2,q3=Q
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[label_vec == 1]=neg_log(preds[label_vec == 1])
    loss_mtx[label_vec == 0]=V[label_vec == 0]*(K[label_vec == 0]*loss1(preds[label_vec == 0],q2)+(1-K[label_vec == 0])*loss2(preds[label_vec == 0],q3))
    main_loss=loss_mtx.mean()
    return main_loss

def hill_loss(preds,label_vec,K,epoch,P=None):
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[label_vec == 1]=neg_log(preds[label_vec == 1])
    loss_mtx[label_vec == 0]=(1.5-preds[label_vec == 0])*preds[label_vec == 0]*preds[label_vec == 0]
    main_loss=loss_mtx.mean()
    return main_loss

def EPR_loss(preds,label_vec,K,epoch,P=None):
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[label_vec == 1]=neg_log(preds[label_vec == 1])
    loss_mtx[label_vec == 0]=0*preds[label_vec == 0]
    main_loss=loss_mtx.mean()
    return main_loss

def weight_loss(preds,label_vec,K,epoch,P=None):
    loss_mtx = torch.zeros_like(preds)
    loss_mtx[label_vec == 1]=8*neg_log(preds[label_vec == 1])
    loss_mtx[label_vec == 0]=0.9*loss2(preds[label_vec == 0],0.99)
    main_loss=loss_mtx.mean()
    return main_loss

LOG_EPSILON = 1e-7

def neg_log(x):
    return - torch.log(x + LOG_EPSILON)

def loss1(x,q):
    return (1 - torch.pow(x, q)) / q

def loss2(x,q):
    return (1 - torch.pow(1-x, q)) / q