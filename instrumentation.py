import numpy as np
import copy
import metrics
import torch

class train_logger:
    
    '''
    An instance of this class keeps track of various metrics throughout
    the training process.
    '''
    
    def __init__(self, params):
        
        self.params = params
        
        # epoch-level objects:
        self.best_stop_metric = -np.Inf
        self.best_epoch = -1
        self.running_loss = 0.0
        self.num_examples = 0
        
        # batch-level objects:
        self.temp_preds = []
        self.temp_true = [] # true labels
        self.temp_obs = [] # observed labels
        self.temp_indices = [] # indices for each example
        self.temp_batch_loss = []
        self.temp_batch_reg = []
        
        # output objects: 
        self.logs = {}
        self.logs['metrics'] = {}
        self.logs['best_preds'] = {}
        self.logs['gt'] ={}
        self.logs['obs'] = {}
        self.logs['targ'] = {}
        self.logs['idx'] = {}
        for field in self.logs:
            for phase in ['train', 'val', 'test']:
                self.logs[field][phase] = {}
    
    def compute_phase_metrics(self, phase, epoch, labels_est):
        
        '''
        Compute and store end-of-phase metrics. 
        '''
        
        self.logs['metrics'][phase][epoch] = {} 
        
        # compute metrics w.r.t. clean ground truth labels:
        metrics_clean = compute_metrics(self.temp_preds, self.temp_true)
        for k in metrics_clean:
            self.logs['metrics'][phase][epoch][k + '_clean'] = metrics_clean[k]
        
        # compute metrics w.r.t. observed labels:
        metrics_observed = compute_metrics(self.temp_preds, self.temp_obs)
        for k in metrics_observed:
            self.logs['metrics'][phase][epoch][k + '_observed'] = metrics_observed[k]
        
        if phase == 'train':
            self.logs['metrics'][phase][epoch]['loss'] = self.running_loss / self.num_examples
            self.logs['metrics'][phase][epoch]['est_labels_k_hat'] = float(np.mean(np.sum(labels_est, axis=1)))
            self.logs['metrics'][phase][epoch]['avg_batch_reg'] = np.mean(self.temp_batch_reg)
        else:
            self.logs['metrics'][phase][epoch]['loss'] = -999
            self.logs['metrics'][phase][epoch]['est_labels_k_hat'] = -999
            self.logs['metrics'][phase][epoch]['avg_batch_reg'] = -999
        self.logs['metrics'][phase][epoch]['preds_k_hat'] = np.mean(np.sum(self.temp_preds, axis=1))
   
    def get_stop_metric(self, phase, epoch, variant):
        
        '''
        Query the stop metric.
        '''
        
        assert variant in ['clean', 'observed']
        return self.logs['metrics'][phase][epoch][self.params['stop_metric'] + '_' + variant]

    def update_phase_data(self, batch):
        
        '''
        Store data from a batch for later use in computing metrics. 
        '''
        
        for i in range(len(batch['idx'])):
            self.temp_preds.append(batch['preds_np'][i, :].tolist())
            self.temp_true.append(batch['label_vec_true'][i, :].tolist())
            self.temp_obs.append(batch['label_vec_obs'][i, :].tolist())
            self.temp_indices.append(int(batch['idx'][i]))
            self.num_examples += 1
        self.temp_batch_loss.append(float(batch['loss_np']))
        self.temp_batch_reg.append(float(batch['reg_loss_np']))
        self.running_loss += float(batch['loss_np'] * batch['image'].size(0))
        
    def reset_phase_data(self):
        
        '''
        Reset for a new phase. 
        '''
        
        self.temp_preds = []
        self.temp_true = []
        self.temp_obs = []
        self.temp_indices = []
        self.temp_batch_reg = []
        self.running_loss = 0.0
        self.num_examples = 0.0
        
    def update_best_results(self, phase, epoch, variant):
        
        '''
        Update the current best epoch info if applicable.
        '''
        
        if phase == 'train':
            return False
        elif phase == 'val':
            assert variant in ['clean', 'observed']
            cur_stop_metric = self.get_stop_metric(phase, epoch, variant)
            if cur_stop_metric > self.best_stop_metric:
                self.best_stop_metric = cur_stop_metric
                self.best_epoch = epoch
                self.logs['best_preds'][phase] = self.temp_preds
                self.logs['gt'][phase] = self.temp_true
                self.logs['obs'][phase] = self.temp_obs
                self.logs['idx'][phase] = self.temp_indices
                return True # new best found
            else:
                return False # new best not found
        elif phase == 'test':
            if epoch == self.best_epoch:
                self.logs['best_preds'][phase] = self.temp_preds
                self.logs['gt'][phase] = self.temp_true
                self.logs['obs'][phase] = self.temp_obs
                self.logs['idx'][phase] = self.temp_indices
            return False
        
    def get_logs(self):
        
        '''
        Return a copy of all log data.
        '''
        
        return copy.deepcopy(self.logs)
    
    def report(self, t_i, t_f, phase, epoch):
        report = '[{}] time: {:.2f} min, loss: {:.3f}, {}: {:.2f}, {}: {:.2f}'.format(
            phase,
            (t_f - t_i) / 60.0,
            self.logs['metrics'][phase][epoch]['loss'],
            self.params['stop_metric'] + '_clean',
            self.get_stop_metric(phase, epoch, 'clean'),
            self.params['stop_metric'] + '_observed',
            self.get_stop_metric(phase, epoch, 'observed'),
            )
        print(report)

        
def compute_metrics(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        average_precision_list.append(metrics.compute_avg_precision(y_true[:, j], y_pred[:, j]))
        
    results['map'] = 100.0 * float(np.mean(average_precision_list))
    ''''
    for k in [1, 3, 5]:
        rec_at_k = np.array([metrics.compute_recall_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        prec_at_k = np.array([metrics.compute_precision_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        results['rec_at_{}'.format(k)] = np.mean(rec_at_k)
        results['prec_at_{}'.format(k)] = np.mean(prec_at_k)
        results['top_{}'.format(k)] = np.mean(prec_at_k > 0)
    '''
    return results

def compute_metrics_openimages(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        observed_idxs = np.where(y_true[:, j] != -1)[0]
        if len(observed_idxs) == 0:
            continue
        average_precision_list.append(metrics.compute_avg_precision(y_true[observed_idxs, j], y_pred[observed_idxs, j]))
        
    results['map'] = 100.0 * float(np.mean(average_precision_list))
    ''''
    for k in [1, 3, 5]:
        rec_at_k = np.array([metrics.compute_recall_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        prec_at_k = np.array([metrics.compute_precision_at_k(y_true[i, :], y_pred[i, :], k) for i in range(num_examples)])
        results['rec_at_{}'.format(k)] = np.mean(rec_at_k)
        results['prec_at_{}'.format(k)] = np.mean(prec_at_k)
        results['top_{}'.format(k)] = np.mean(prec_at_k > 0)
    '''
    return results

def compute_aps_openimages(y_pred, y_true):
    
    '''
    Given predictions and labels, compute a few metrics.
    '''
    
    num_examples, num_classes = np.shape(y_true)
    
    results = {}
    average_precision_list = []
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # y_true = np.array(y_true == 1, dtype=np.float32) # convert from -1 / 1 format to 0 / 1 format
    for j in range(num_classes):
        observed_idxs = np.where(y_true[:, j] != -1)[0]
        if len(observed_idxs) == 0:
            continue
        average_precision_list.append(metrics.compute_avg_precision(y_true[observed_idxs, j], y_pred[observed_idxs, j]))

    return np.array(average_precision_list)

def prob_stat(preds,label_vec_obs,label_vec_true,num_bins):
    pos1 = (label_vec_obs == 0) & (label_vec_true == 1)
    pos2 = (label_vec_obs == 0) & (label_vec_true == 0)
    
    elements1 = preds[pos1]
    elements2 = preds[pos2]
    dis=wasserstein_distance(elements1, elements2)
    counts1, bin_edges1 = torch.histogram(elements1, bins=num_bins)
    counts2, bin_edges2 = torch.histogram(elements2, bins=num_bins)

    prob=counts1/(counts1+counts2+1e-9)
    
    return prob,counts1,counts2,bin_edges1[1:],dis

import matplotlib.pyplot as plt
plt.style.use("ggplot")
import matplotlib.patches as mpatches
def scatter1(bins,prob,epoch):
    save_path=f'results/plot18/{epoch}'
    plt.scatter(bins,prob)
    plt.savefig(save_path,dpi=400)
    plt.clf()

def hist1(bins1,bins2,counts1, counts2, epoch, flag):
    bins1=bins1.tolist()
    counts1=counts1.tolist()
    bins2=bins2.tolist()
    counts2=counts2.tolist()
    fig, axs = plt.subplots(2, 1, figsize=(5,4), gridspec_kw={'hspace': 0})

    # Plot first histogram in blue
    axs[0].bar(bins1, counts1, width=1/len(bins1), alpha=0.7, color='lightgreen')
    axs[0].set_xlabel('Label Confidence')
    axs[0].set_ylabel('Counts', fontsize=12)
    axs[0].xaxis.set_tick_params(labelbottom=False)  # Hide x-axis labels for the top histogram

    # Plot second histogram in orange
    axs[1].bar(bins2, counts2, width=1/len(bins2), alpha=0.7, color='lightcoral')
    axs[1].set_xlabel('Label Confidence',fontsize=15, fontweight='bold')
    axs[1].set_ylabel('Counts', fontsize=12)
    positive_patch = mpatches.Patch(color='lightgreen', label='Positive')
    negative_patch = mpatches.Patch(color='lightcoral', label='Negative')
    axs[0].legend(handles=[positive_patch, negative_patch], loc='upper center')
    for ax in axs:
        for label in ax.get_yticklabels():
            label.set_fontsize(8)
            label.set_style('italic')
    # Tight layout to ensure no overlap
    plt.tight_layout()


    # Annotate every 5th bin with its count
    # for i in range(len(counts)):
    #     if i % 5 == 0:  # Every 5th bin
    #         plt.text(bins[i], counts[i] + 0.05 * max(counts), str(int(counts[i])), ha='center')

    save_path = f'results/plot18/{epoch}{flag}'
    plt.savefig(save_path,dpi=800)
    plt.clf()
from scipy.stats import wasserstein_distance
def cal(counts1,counts2):
    return wasserstein_distance(counts1, counts2)