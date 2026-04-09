import json
import os


class Config:
    def __init__(self):

        # =========================================================================
        # NECESSARY PARAMETERS
        # =========================================================================

        # Absolute path to the ChromTransfer directory on your machine
        self.YOUR_PATH_TO_ChromTransfer = '/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub'

        # Absolute path to the directory to save the model and predictions
        self.output_dir = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/3.predict"

        self.tf = "SOX2"  # TF to predict (must be uppercase)

        self.source_species = "mm10"  # or "hg38"
        self.target_species = "hg38"  # or "mm10"

        # Absolute path to the BED-format peak file for TF binding sites in the source species
        self.peak_file_source = "/mnt/Storage/home/wangyiman/crossSpecies_TF_bindingSite_prediction/bin_for_GitHub/demo/1.preprocess_data/GSM307138.narrowPeak.bed"

        # =========================================================================
        # OPTIONAL PARAMETERS
        # =========================================================================

        self.random_seed = 9999  # Random seed for preprocessing and training

        # -------------------------------------------------------------------------
        # Preprocessing settings
        # -------------------------------------------------------------------------

        # Path to the BED-format peak file for TF binding sites in the target species.
        # Leave as "" if you do not have target-species peak data.
        self.peak_file_target = ""

        self.genome_bin_file = {
            'hg38': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "regions", "hg38_500_50_noblack_k36_noN_window_regionNumber.bed"),
            'mm10': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "regions", "mm10_500_50_noblack_k36_noN_window_regionNumber.bed"),
        }

        self.test_chromosome  = "chr2"   # Chromosome held out for testing
        self.valid_chromosome = "chr1"   # Chromosome held out for validation.
                                         # 100k regions are used for validation;
                                         # the rest are merged into the training set.
        self.valid_chromosome_region_num = 100000  # Number of regions used for validation
        self.max_epoch_num = 100                   # Maximum number of training epochs

        # -------------------------------------------------------------------------
        # Training settings
        # -------------------------------------------------------------------------

        self.model = "ChromTransfer-Reg"  # or "ChromTransfer-Cons" or "ChromTransfer-Base"

        self.DNA_file = {
            'hg38': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "DNA", "hg38_DNA_500bpBin50bpStep_region.h5"),
            'mm10': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "DNA", "mm10_DNA_500bpBin50bpStep_region.h5"),
        }

        self.FUNCODE_file = {
            'hg38': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "FUNCODE", "hg38_FUNCODE_avgScore_50050.pkl"),
            'mm10': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "FUNCODE", "mm10_FUNCODE_avgScore_50050.pkl"),
        }

        self.cobindingTFs_mask_file = os.path.join(
            self.YOUR_PATH_TO_ChromTransfer, "data", "Regulatory", "TF_h5Column_mask_dict_prepared.json"
        )
        with open(self.cobindingTFs_mask_file, 'r') as f:
            cobindingTFs_mask_dict = json.load(f)

        if self.tf not in cobindingTFs_mask_dict:
            # Generate mask on-the-fly for TFs not covered by the pre-built dictionary
            from utils.generate_mask_json_for_ChromTransferReg import generate_mask_json_for_ChromTransferReg
            cobindingTFs_mask_dict = generate_mask_json_for_ChromTransferReg(self.YOUR_PATH_TO_ChromTransfer, self.tf, self.output_dir)

        self.cobindingTFs_mask = cobindingTFs_mask_dict[self.tf]

        self.cobindingTFs_file = {
            'hg38': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "Regulatory", "hg38_Reg_signal_matrix.hdf5"),
            'mm10': os.path.join(self.YOUR_PATH_TO_ChromTransfer, "data", "Regulatory", "mm10_Reg_signal_matrix.hdf5"),
        }

        self.gpu         = "0"    # GPU index to use for training
        self.batch_size  = 1024   # Batch size for training
        self.lr          = 1e-4   # Learning rate
        self.num_workers = 32     # Number of DataLoader worker processes
        self.num_epochs  = 100    # Maximum number of training epochs

        # -------------------------------------------------------------------------
        # Prediction settings
        # -------------------------------------------------------------------------

        self.predict_batch_size  = 4096   # Batch size for prediction
        self.predict_num_workers = 32     # Number of DataLoader worker processes for prediction
        self.threshold_bin_width = 0.0001 # Threshold bin width (smaller = more accurate but slower)
        self.threshold_FDR_cutoff = 0.01  # FDR cutoff for peak calling

    def __str__(self):
        return (
            f"Config(\n"
            f"  output_dir                  = {self.output_dir}\n"
            f"  source_species              = {self.source_species}\n"
            f"  target_species              = {self.target_species}\n"
            f"  random_seed                 = {self.random_seed}\n"
            f"  peak_file_source            = {self.peak_file_source}\n"
            f"  peak_file_target            = {self.peak_file_target}\n"
            f"  tf                          = {self.tf}\n"
            f"  genome_bin_file             = {self.genome_bin_file}\n"
            f"  test_chromosome             = {self.test_chromosome}\n"
            f"  valid_chromosome            = {self.valid_chromosome}\n"
            f"  valid_chromosome_region_num = {self.valid_chromosome_region_num}\n"
            f"  max_epoch_num               = {self.max_epoch_num}\n"
            f"  model                       = {self.model}\n"
            f"  DNA_file                    = {self.DNA_file}\n"
            f"  FUNCODE_file                = {self.FUNCODE_file}\n"
            f"  cobindingTFs_mask_file      = {self.cobindingTFs_mask_file}\n"
            f"  gpu                         = {self.gpu}\n"
            f"  batch_size                  = {self.batch_size}\n"
            f"  lr                          = {self.lr}\n"
            f"  num_workers                 = {self.num_workers}\n"
            f"  num_epochs                  = {self.num_epochs}\n"
            f"  predict_batch_size          = {self.predict_batch_size}\n"
            f"  predict_num_workers         = {self.predict_num_workers}\n"
            f"  threshold_bin_width         = {self.threshold_bin_width}\n"
            f"  threshold_FDR_cutoff        = {self.threshold_FDR_cutoff}\n"
            f")"
        )
