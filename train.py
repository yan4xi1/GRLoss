import numpy as np
from sklearn import metrics
import torch
import datasets
import models
from instrumentation import compute_metrics,prob_stat,scatter1,hist1,cal
import losses
import datetime
import os
from tqdm import tqdm

def run_train(P):
    dataset = datasets.get_data(P)
    dataloader = {}
    for phase in ['train', 'val', 'test']:
        dataloader[phase] = torch.utils.data.DataLoader(
            dataset[phase],
            batch_size = P['bsize'],
            shuffle = phase == 'train',
            sampler = None,
            num_workers = P['num_workers'],
            drop_last = False,
            pin_memory = True
        )

    model = models.ImageClassifier(P)
    feature_extractor_params = [param for param in list(model.feature_extractor.parameters()) if param.requires_grad]
    linear_classifier_params = [param for param in list(model.linear_classifier.parameters()) if param.requires_grad]
    
    opt_params = [
        {'params': feature_extractor_params, 'lr' : P['lr']},
        {'params': linear_classifier_params, 'lr' : P['lr_mult'] * P['lr']},
        
    ]

    if P['is_train'] == True:
        label_estimator=models.LabelEstimator(P)
        label_estimator_params = [param for param in list(label_estimator.parameters()) if param.requires_grad]
        opt_params1 = [
            {'params': label_estimator_params, 'lr' : 1e-1},
        ]
        optimizer1 = torch.optim.SGD(opt_params1, lr=1e-1, momentum=0.9, weight_decay=0.001)
        label_estimator.to(device)

    if P['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(opt_params, lr=P['lr'])
    elif P['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(opt_params, lr=P['lr'], momentum=0.9, weight_decay=0.001)
    
    # training loop
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    bestmap_val = 0
    bestmap_test = 0
    val_loss_lst=[]
    dis_lst=[]
    for epoch in range(1, P['num_epochs']+1):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                n=P['linear_init']
                if epoch <= n:
                    for param in model.feature_extractor.parameters():
                        param.requires_grad = False
                else:
                    for param in model.feature_extractor.parameters():
                        param.requires_grad = True
                #beta
                w_0,w_max,b_0,b_max=P['beta']
                w=w_0+(w_max-w_0)*max((epoch-n-1),0)/(P['num_epochs']-n)
                b=b_0+(b_max-b_0)*max((epoch-n-1),0)/(P['num_epochs']-n)
                #alpha
                mu_0,sigma_0,mu_max,sigma_max=P['alpha']
                mu=mu_0+(mu_max-mu_0)*max((epoch-n),0)/(P['num_epochs']-n)
                sigma=sigma_0+(sigma_max-sigma_0)*max((epoch-n),0)/(P['num_epochs']-n)  
                #q2q3
                Q=P['q2q3']             
                k_function = models.KFunction(w, b)
                
                print(f'w:{w},b:{b}')
                print(f'mu:{mu};sigma:{sigma}')
                print(f'q2:{Q[0]};q3:{Q[1]}')
            else:
                model.eval()
                if P['is_train'] == True:
                    label_estimator.eval()
                y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
                y_true = np.zeros((len(dataset[phase]), P['num_classes']))
                y_obs = np.zeros((len(dataset[phase]), P['num_classes']))
                batch_stack = 0

            with torch.set_grad_enabled(phase == 'train'):
                for batch in tqdm(dataloader[phase]):
                    # Move data to GPU
                    image = batch['image'].to(device, non_blocking=True)
                    label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
                    label_vec_true = batch['label_vec_true'].clone().numpy()
                    idx = batch['idx']

                    # Forward pass
                    optimizer.zero_grad()
                    if P['is_train'] == True:
                        optimizer1.zero_grad()
                    logits = model(image)
                    if torch.isnan(logits).any():
                        print(logits) 
                    if logits.dim() == 1:
                        logits = torch.unsqueeze(logits, 0)
                    preds = torch.sigmoid(logits)
                    preds1=preds.detach()
                    if P['is_train'] == True:
                        K=label_estimator(preds1)
                    else:
                        K = k_function(preds1) 
                    V=models.VFunction(preds1,mu,sigma)
                    if phase == 'train':
                        loss=losses.GR_loss(preds,label_vec_obs,K,V,Q,epoch)
                        loss.backward()
                        optimizer.step()
                        if P['is_train'] == True:
                            optimizer1.step()
                    else:
                        preds_np = preds.cpu().numpy()
                        label_vec_obs_np=label_vec_obs.cpu().numpy()
                        this_batch_size = preds_np.shape[0]
                        y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
                        y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
                        y_obs[batch_stack : batch_stack+this_batch_size] = label_vec_obs_np
                        
                        y_pred1=torch.Tensor(y_pred)
                        y_true1=torch.LongTensor(y_true)
                        y_obs1=torch.LongTensor(y_obs)

                        batch_stack += this_batch_size
        metrics = compute_metrics(y_pred, y_true)

        map_val = metrics['map']
                
        print(f"Epoch {epoch} : val mAP {map_val:.3f}")
        val_loss_lst.append(map_val)
        # prob,counts1,counts2,bins,dis=prob_stat(y_pred1,y_obs1,y_true1,num_bins=100)
        # scatter1(bins,prob,epoch)
        # hist1(bins,bins,counts1,counts2,epoch,flag=9)
        # dis=cal(counts1,counts2)
        # print(dis)
        # dis_lst.append(dis)
        del y_pred
        del y_true
        del y_obs
        if bestmap_val < map_val:
            bestmap_val = map_val
            bestmap_epoch = epoch
            
            print(f'Saving model weight for best val mAP {bestmap_val:.3f}')
            path = os.path.join(P['save_path'], 'bestmodel.pt')
            torch.save((model.state_dict(), P), path)
        
        elif bestmap_val - map_val > 2:
            print('Early stopped.')
            break
    # Test phase

    model_state, _ = torch.load(path)
    model.load_state_dict(model_state)

    phase = 'test'
    model.eval()
    if P['is_train'] == True:
        label_estimator.eval()
    y_pred = np.zeros((len(dataset[phase]), P['num_classes']))
    y_true = np.zeros((len(dataset[phase]), P['num_classes']))
    batch_stack = 0
    with torch.set_grad_enabled(phase == 'train'):
        for batch in tqdm(dataloader[phase]):
            # Move data to GPU
            image = batch['image'].to(device, non_blocking=True)
            label_vec_obs = batch['label_vec_obs'].to(device, non_blocking=True)
            label_vec_true = batch['label_vec_true'].clone().numpy()
            idx = batch['idx']

            # Forward pass
            optimizer.zero_grad()
            if P['is_train'] == True:
                optimizer1.zero_grad()
            logits = model(image)
                   
            if logits.dim() == 1:
                logits = torch.unsqueeze(logits, 0)
            preds = torch.sigmoid(logits)
               
            preds_np = preds.cpu().numpy()
            this_batch_size = preds_np.shape[0]
            y_pred[batch_stack : batch_stack+this_batch_size] = preds_np
            y_true[batch_stack : batch_stack+this_batch_size] = label_vec_true
            batch_stack += this_batch_size

    metrics = compute_metrics(y_pred, y_true)
    map_test = metrics['map']
    print('val_map :',[round(x,2) for x in val_loss_lst])
    print(f'Test mAP : {map_test:.3f} when trained until epoch {bestmap_epoch}')
    # print([round(x,3) for x in dis_lst])
    