import os
import pandas as pd
import glob
import subprocess

from config import Config

config = Config()

os.makedirs(config.output_dir, exist_ok=True)
os.chdir(config.output_dir)
output_file = f'{config.source_species}_label.bed'
peak_fi = config.peak_file_source
print("Peak file for preprocessing", peak_fi)

print(" Start preprocessing the peak file...")
print(" Step 1: Intersect the peak file with the genome bin file")
cmd = f'''bedtools intersect -a {config.genome_bin_file[config.source_species]} -b {config.peak_file_source} -c | awk 'BEGIN{{FS="\\t"; OFS="\\t"}} {{print $1, $2, $3, $4, ($5 >= 1)}}' > {output_file}'''
print(cmd)
subprocess.run(
    cmd,
    shell=True
)

print(" Step 2: Read the output file and split the data into train, valid and test")
label_df = pd.read_csv(output_file, sep='\t', header=None, names=['chrom','start','end','region_num','label'])
train_df = label_df[(label_df['chrom'] != config.test_chromosome) & (label_df['chrom'] != config.valid_chromosome)]
chr1_df = label_df[label_df['chrom'] == config.valid_chromosome]
valid_df = chr1_df.sample(n=config.valid_chromosome_region_num, random_state=config.random_seed)
chr1_df_others = chr1_df.drop(valid_df.index)
train_df = pd.concat([train_df, chr1_df_others])
test_df = label_df[label_df['chrom'] == config.test_chromosome]

train_df[['region_num', 'label']].to_pickle(f'{config.output_dir}/{config.source_species}_train.pkl')
valid_df[['region_num', 'label']].to_pickle(f'{config.output_dir}/{config.source_species}_valid.pkl')
test_df[['region_num', 'label']].to_pickle(f'{config.output_dir}/{config.source_species}_test.pkl')

valid_df[['region_num', 'label']].to_csv(f'{config.output_dir}/{config.source_species}.{config.valid_chromosome}_random{config.valid_chromosome_region_num}.txt', sep='\t', index=None)
test_df[['region_num', 'label']].to_csv(f'{config.output_dir}/{config.source_species}.{config.test_chromosome}.txt', sep='\t', index=None)

train_df_neg = train_df.loc[train_df['label'] == 0].sample(frac=1, random_state=config.random_seed)
train_df_pos = train_df.loc[train_df['label'] == 1]
        
if train_df_pos.shape[0] == 0 :
    print(f"{config.peak_file_source} has no positive samples")
    exit()
        
possible_epoch_num = int(train_df_neg.shape[0] / train_df_pos.shape[0]) + 1
for epoch in range(possible_epoch_num) :
    end_row = min((epoch + 1) * train_df_pos.shape[0], train_df_neg.shape[0])
    pd.concat(
        [
            train_df_pos, 
            train_df_neg.iloc[epoch * train_df_pos.shape[0]: end_row]
            ]
        ).sample(frac=1)[['region_num', 'label']].to_csv(f'{config.source_species}.chrOthers_epoch{epoch+1}.txt', sep='\t', index=None)
    
    if epoch > config.max_epoch_num :
        break
    
    
print(f"{config.peak_file_source} processed")


if config.peak_file_target != "" :
    print(" Start preprocessing the peak file for target species...")
    print(" Step 1: Intersect the peak file with the genome bin file")
    subprocess.run(
        f'''bedtools intersect -a {config.genome_bin_file[config.target_species]} -b {config.peak_file_target} -c | awk 'BEGIN{{FS="\\t"; OFS="\\t"}} {{print $1, $2, $3, $4, ($5 >= 1)}}' > {output_file}''',
        shell=True
    )
    
    print(" Step 2: Read the output file and select test chromosome as test dataset")
    label_df = pd.read_csv(output_file, sep='\t', header=None, names=['chrom','start','end','region_num','label'])
    label_df_test = label_df[label_df['chrom'] == config.test_chromosome]
    label_df[['region_num', 'label']].to_csv(f'{config.output_dir}/{config.target_species}.chrTotal.txt', sep='\t', index=None)
    label_df_test[['region_num', 'label']].to_csv(f'{config.output_dir}/{config.target_species}.{config.test_chromosome}.txt', sep='\t', index=None)
    
    print(f"{config.peak_file_target} processed")

else :
    ### only prepare the total regions for target species prediction
    ### set "label" to be 0 for all regions
    label_df = pd.read_csv(f'{config.genome_bin_file[config.target_species]}', sep='\t', header=None, names=['chrom','start','end','region_num'])
    label_df['label'] = 0
    label_df[['region_num', 'label']].to_csv(f'{config.output_dir}/{config.target_species}.chrTotal.txt', sep='\t', index=None)
    
    