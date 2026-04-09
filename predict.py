### export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/jupyter_kernel/lib

import os
import glob
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import subprocess

### load model and dataset
from model import ChromTransfer_Base, ChromTransfer_Cons, ChromTransfer_Reg
from dataset import SupervisedDataset
from utils.summary_FDR_to_50bp import process_prediction_file
from config import Config

### ====== System settings ====== ###
config = Config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
torch.set_num_threads(config.predict_num_workers)
torch.multiprocessing.set_sharing_strategy('file_system')
os.chdir(config.output_dir)
device = torch.device(f'cuda:0')
torch.set_num_threads(config.num_workers)


### ====== Functions ====== ###
def predict(predict_loader):
    
    if config.model == "ChromTransfer-Base":
        model = ChromTransfer_Base()
    elif config.model == "ChromTransfer-Cons":
        model = ChromTransfer_Cons()
    elif config.model == "ChromTransfer-Reg":
        model = ChromTransfer_Reg(cis_feature_num=sum(config.cobindingTFs_mask))
    
    model.to(device)
    model.load_state_dict(torch.load(f"best_model.pth"))
    model.eval()
    all_predictions = []
    all_region_nums = []
    with torch.no_grad():
        for d in tqdm(predict_loader):
            one_hot_seqs = d['one_hot_seq'].float().to(device)
            datas_FUNCODE = d['data_FUNCODE'].float().to(device)
            cobindingTFs_data = d['cobindingTFs'].float().to(device)
            region_nums = d['region_num']
            outputs_classifier = model(one_hot_seqs, datas_FUNCODE, cobindingTFs_data)
            all_predictions.extend(outputs_classifier.view(-1).cpu().numpy())
            all_region_nums.extend(region_nums)

    all_predictions = np.array(all_predictions)
    all_region_nums = np.array(all_region_nums)
    all_predictions_df = pd.DataFrame({'region_num': all_region_nums, 'prediction': all_predictions})
    return all_predictions_df


### function to calculate FDR
def compute_fdr_vectorized(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    pred_pos = y_prob[:, None] >= thresholds[None, :]   # shape: (n_samples, n_thresholds)

    tp = np.sum(pred_pos & (y_true[:, None] == 1), axis=0)
    fp = np.sum(pred_pos & (y_true[:, None] == 0), axis=0)
    fn = np.sum((~pred_pos) & (y_true[:, None] == 1), axis=0)
    tn = np.sum((~pred_pos) & (y_true[:, None] == 0), axis=0)

    called_positive = tp + fp
    precision = np.divide(tp, called_positive, out=np.full_like(tp, np.nan, dtype=float), where=called_positive > 0)
    recall = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fn) > 0)
    fdr = np.divide(fp, called_positive, out=np.full_like(fp, np.nan, dtype=float), where=called_positive > 0)

    return pd.DataFrame({
        "threshold": thresholds,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "FDR": fdr
    })


# ### predict TF-binding sites in target species
# print(f"start predicting {config.tf} (cross)")
# predict_dataset = SupervisedDataset(
#     f'{config.output_dir}/{config.target_species}.chrTotal.txt',
#     config.FUNCODE_file[config.target_species],
#     config.DNA_file[config.target_species],
#     config.cobindingTFs_file[config.target_species],
#     config.cobindingTFs_mask
# )
# predict_loader = DataLoader(predict_dataset, batch_size=config.predict_batch_size, num_workers=config.predict_num_workers)
# pred_total_cross_df = predict(predict_loader)
# pred_total_cross_df.to_csv(f"predictions_{config.target_species}.txt", index=False)

# pred_total_cross_df = pd.read_csv(f"predictions_{config.target_species}.txt") ######

# ### predict TF-binding sites in source species (chr2). This is for FDR calculation.
# pred_chr2_self_df = pd.read_csv(f"predictions_{config.source_species}Chr2.txt")


# ### calculate FDR according to different thresholds.
# true_chr2_self_df = pd.read_csv(f'{config.source_species}.chr2.txt', header=0, sep = "\t")
# self_chr2_df = pd.concat([true_chr2_self_df.set_index('region_num'), pred_chr2_self_df.set_index('region_num')], axis = 1)
# fdr_df = compute_fdr_vectorized(true_chr2_self_df['label'], pred_chr2_self_df['prediction'], np.arange(1 - config.threshold_bin_width, -config.threshold_bin_width, -config.threshold_bin_width)) ### 0.9999, 0.9998, 0.9997, ... 0.0000
# fdr_df['threshold_digit4'] = np.round(fdr_df['threshold'], 4)
# fdr_df['FDR'].replace(0, config.threshold_bin_width, inplace=True) ### for the thresholds make FDR equals to 0
# fdr_df['FDR'].replace(np.nan, config.threshold_bin_width, inplace=True) ### for the thresholds over max(prob)
# fdr_df.to_csv(f'{config.source_species}_chr2_threshold_FDR.csv', header=True, index=False)

# ### reflect the FDR on the total predictions in target species.
# pred_total_cross_df['prediction_digit4'] = np.round(pred_total_cross_df['prediction'], 4)
# pred_total_cross_df['FDR'] = pred_total_cross_df['prediction_digit4'].replace(1, 1-config.threshold_bin_width).map(fdr_df.set_index('threshold_digit4')['FDR'])
# pred_total_cross_df['-10lg(FDR)'] = -10 * np.log10(pred_total_cross_df['FDR'])
# pred_total_cross_df.to_csv(f'{config.target_species}_total_threshold_FDR.csv', header=True, index=False)



# ### summarize the predictions into 50bp bins.
# bed_df_total = pd.read_csv(config.genome_bin_file[config.target_species], sep="\t", header=None, names=['chr', 'start', 'end', 'regionNumber']).set_index('regionNumber')
# bed_fi = f'{config.target_species}_total_threshold_FDR.csv'
# msg = process_prediction_file(bed_fi, bed_df_total, config)
# print(msg)

# ### convert to BigWig file.
# bed_df = pd.read_csv(f'FDR_{config.target_species}_50bp_bins.bed', sep='\t', header=None, names=['chr', 'start', 'end', 'score'])
# # Build the WIG position lines: 1-based start + score
# bed_df['wig_line'] = (bed_df['start'] + 1).astype(str) + ' ' + bed_df['score'].astype(str)
# wig_lines = []
# current_chrom = None
# for chrom, group in bed_df.groupby('chr', sort=False):
#     if chrom != current_chrom:
#         current_chrom = chrom
#         wig_lines.append(f"variableStep chrom={chrom} span=50")
#     wig_lines.extend(group['wig_line'].tolist())
# out_file = f'FDR_{config.target_species}_50bp_bins.wig'
# with open(out_file, 'w') as f:
#     f.write('\n'.join(wig_lines) + '\n')


### call peaks based on the FDR-reflected predictions. FDR cutoff is 0.2.

cmd = f"macs2 bdgpeakcall -i FDR_{config.target_species}_50bp_bins.bed -c 6.9897 -o {config.target_species}_narrowPeak.bed"
print(f"MACS2 callpeak command: {cmd}")

subprocess.run(cmd, shell=True)

### remove the first line in {config.target_species}_narrowPeak.bed
with open(f"{config.target_species}_narrowPeak.bed", "r") as f:
    lines = f.readlines()
with open(f"{config.target_species}_narrowPeak.bed", "w") as f:
    f.writelines(lines[1:])
    