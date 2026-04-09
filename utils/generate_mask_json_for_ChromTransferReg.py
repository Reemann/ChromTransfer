import pandas as pd
import os
import json
import glob


def generate_mask_json_for_ChromTransferReg(YOUR_PATH_TO_ChromTransfer, tf, output_dir):
    ens_protein_id_df_hg38_fi = os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "ensemblProteinID_2_geneName_hg38.txt")    
    ens_protein_id_df_mm10_fi = os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "ensemblProteinID_2_geneName_mm10.txt")
    ens_protein_id_df_hg38 = pd.read_csv(ens_protein_id_df_hg38_fi, sep = "\t").dropna()
    ens_protein_id_df_mm10 = pd.read_csv(ens_protein_id_df_mm10_fi, sep = "\t").dropna()
    
    tf_ensPid_list = [
        [
            tf, 
            ('9606.' + ens_protein_id_df_hg38[ens_protein_id_df_hg38["Gene name"] == tf]["Protein stable ID"]).tolist(), 
            ('10090.' + ens_protein_id_df_mm10[ens_protein_id_df_mm10["Gene name"].str.upper() == tf]["Protein stable ID"]).tolist(), 
        ] for tf in [tf] if (tf in ens_protein_id_df_hg38["Gene name"].tolist()) and (tf in ens_protein_id_df_mm10["Gene name"].str.upper().tolist())
        ]
    
    if len(tf_ensPid_list) == 0:
        raise ValueError(f"No ENSEMBL protein ID found for {tf} in hg38 and mm10. Please select a valid TF name as config.tf.")
    
    chromatin_context_ls = [
        'ATAC-seq','DNase-seq','FAIRE-seq','H2AZ','H2Bub','H3K27ac','H3K27me3',
        'H3K36me3','H3K4me1','H3K4me2','H3K4me3','H3K56ac','H3K79me2',
        'H3K9K14ac','H3K9ac','H3K9me2','H3K9me3','H3ac','H4K16ac',
        'H4K20me1','H4K20me3','H4K5ac'
        ]
    
    ensembl_id_dict = {}
    ensembl_id_dict['hg38'] = ens_protein_id_df_hg38[['Protein stable ID', 'Gene name']].set_index('Protein stable ID')['Gene name'].str.upper().to_dict()
    ensembl_id_dict['mm10'] = ens_protein_id_df_mm10[['Protein stable ID', 'Gene name']].set_index('Protein stable ID')['Gene name'].str.upper().to_dict()
    
    mat_column_name = pd.read_csv(os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "cobindingTF_chromatinContext_ls.txt"), header = 0)['Factor']
    
    cobindingTFs_mask_dict = {}
    tf_interaction_number_ls = []
    tf_interaction_tf_dict = {}
    tf, tf_h_ls, tf_m_ls = tf_ensPid_list[0]
    
    ### interaction data from STRING
    string_ppi_files = {
        'hg38': os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "STRING", "9606.protein.links.full.v12.0.txt"),
        'mm10': os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "STRING", "10090.protein.links.full.v12.0.txt")
    }

    string_ppi_df_total = {}

    for species in ['hg38', 'mm10'] :
        string_ppi_df_total[species] = pd.read_csv(string_ppi_files[species], sep=' ')[['protein1', 'protein2', 'combined_score']]        
        string_ppi_df_total[species]['symbol1'] = string_ppi_df_total[species]['protein1'].str.split('.').str[1].map(ensembl_id_dict[species]).str.upper()
        string_ppi_df_total[species]['symbol2'] = string_ppi_df_total[species]['protein2'].str.split('.').str[1].map(ensembl_id_dict[species]).str.upper()
    
    
    tf_interaction_hg38 = string_ppi_df_total['hg38'][(string_ppi_df_total['hg38']['protein1'].isin(tf_h_ls)) & (string_ppi_df_total['hg38']['combined_score'] >= 700)]['symbol2'].unique().tolist()
    tf_interaction_mm10 = string_ppi_df_total['mm10'][(string_ppi_df_total['mm10']['protein1'].isin(tf_m_ls)) & (string_ppi_df_total['mm10']['combined_score'] >= 700)]['symbol2'].unique().tolist()
    cobinding_tf_list1 = set(tf_interaction_hg38) & set(tf_interaction_mm10)   
    cobinding_tf_list1 = [x for x in cobinding_tf_list1 if x != tf]      
    cobinding_tf_list1 = pd.Series(cobinding_tf_list1).dropna().unique().tolist()
    
    
    ### cobinding data from ChIP-Atlas
    df_tf_total = pd.DataFrame()
    chipatlas_fi_ls = glob.glob(os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "ChIP_Atlas", f"{tf}.*.tsv")) + glob.glob(os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "ChIP_Atlas", f"{tf.upper()}.*.tsv"))
    if len(chipatlas_fi_ls) > 0 :
        for fi in chipatlas_fi_ls :
            df = pd.read_csv(fi, sep = "\t")
            df = df[['Experiment', 'Cell_subclass', 'Protein', f'{tf}|Average']]
            df = df[df[f'{tf}|Average'] > 0]
            df_tf_total = pd.concat([df_tf_total, df])

        cobinding_tf_list2 = df_tf_total[df_tf_total[f'{tf}|Average'] > 4.5]['Protein'].value_counts().head(5).index.tolist()
        cobinding_tf_list2 = [x for x in cobinding_tf_list2 if x != tf]      
    else :
        cobinding_tf_list2 = []
    
    ### Cap-SELEX data
    cap_df_interaction = pd.read_csv(os.path.join(YOUR_PATH_TO_ChromTransfer, "data", "cobinding_TF_source", "CAP_SELEX", "CAPSELEX_TF_TF_interaction.csv"))
    cobinding_tf_list3 = cap_df_interaction[cap_df_interaction['TF1'] == tf.upper()]['TF2'].unique().tolist()
    
    ### combine all cobinding TFs
    cobinding_tf_list = list(set(cobinding_tf_list1) | set(cobinding_tf_list2) | set(cobinding_tf_list3))
    cobinding_tf_list = pd.Series(cobinding_tf_list).dropna().unique().tolist()
    
    ### set cobinding TFs and non-TFs as True
    column_mask = mat_column_name.isin(cobinding_tf_list) + mat_column_name.isin(chromatin_context_ls)
    ### exclude the TF itself
    column_mask = [x if mat_column_name[i] != tf else False for i, x in enumerate(column_mask)]
    cobindingTFs_mask_dict[tf] = column_mask
    
    tf_interaction_number_ls.append(sum(column_mask) - mat_column_name.isin(chromatin_context_ls).sum()) #####
    tf_interaction_tf_dict[tf] = cobinding_tf_list
    
    with open(os.path.join(output_dir, f"TF_h5Column_mask_dict_{tf}.json"), 'w') as f:
        json.dump(cobindingTFs_mask_dict, f)

    with open(os.path.join(output_dir, f"TF_interaction_tf_dict_{tf}.json"), 'w') as f:
        json.dump(tf_interaction_tf_dict, f)
        
    tf_interaction_number_df = pd.DataFrame(tf_interaction_number_ls, index = cobindingTFs_mask_dict.keys(), columns = ['TF_interaction_number'])
    tf_interaction_number_df.to_csv(os.path.join(output_dir, f"TF_interaction_number_{tf}.csv"), index = True)

    
    return cobindingTFs_mask_dict