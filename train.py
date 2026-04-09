import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import roc_auc_score, precision_score, recall_score, matthews_corrcoef, f1_score, precision_recall_curve, auc
from tqdm import tqdm
import pandas as pd
import numpy as np

### load model and dataset
from model import ChromTransfer_Base, ChromTransfer_Cons, ChromTransfer_Reg
from dataset import SupervisedDataset
from config import Config


### ====== System settings ====== ###
config = Config()
os.makedirs(config.output_dir, exist_ok=True)
os.chdir(config.output_dir)
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
device = torch.device(f'cuda:0')
torch.set_num_threads(config.num_workers)
torch.multiprocessing.set_sharing_strategy('file_system')


### ====== Functions ====== ###
def calculate_accuracy(outputs, labels):
    predicted = (outputs >= 0.5).float()
    correct = (predicted == labels.unsqueeze(1))
    accuracy = sum(correct)
    return accuracy.item()


def validate_model(model, val_loader):
    model.eval()
    valid_loss = 0.0
    total_accuracy = 0.0
    all_labels = []
    all_predictions = []
    criterion = nn.BCELoss()
    with torch.no_grad():
        for d in tqdm(val_loader):
            one_hot_seqs = d['one_hot_seq'].float().to(device)
            labels_classifier = d['label_classifier'].float().to(device)
            datas_FUNCODE = d['data_FUNCODE'].float().to(device)
            cobindingTFs_data = d['cobindingTFs'].float().to(device)
            outputs_classifier = model(one_hot_seqs, datas_FUNCODE, cobindingTFs_data)
            loss = criterion(outputs_classifier, labels_classifier.unsqueeze(1))
            accuracy = calculate_accuracy(outputs_classifier, labels_classifier)
            total_accuracy += accuracy
            valid_loss += loss.item() * one_hot_seqs.size(0)
            all_labels.extend(labels_classifier.view(-1).cpu().numpy())
            all_predictions.extend(outputs_classifier.view(-1).cpu().numpy())

    valid_loss = valid_loss / len(val_loader.dataset)
    valid_accuracy = total_accuracy / len(val_loader.dataset)

    roc_auc = roc_auc_score(all_labels, all_predictions)
    binary_predictions = [1 if x >= 0.5 else 0 for x in all_predictions]
    precision = precision_score(all_labels, binary_predictions)
    recall = recall_score(all_labels, binary_predictions)
    mcc = matthews_corrcoef(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)

    precision2, recal2, _ = precision_recall_curve(all_labels, all_predictions)
    auprc = auc(recal2, precision2)

    return valid_loss, valid_accuracy, roc_auc, precision, recall, mcc, f1, auprc


def test_model(test_loader):
    if config.model == "ChromTransfer-Base":
        model = ChromTransfer_Base()
    elif config.model == "ChromTransfer-Cons":
        model = ChromTransfer_Cons()
    elif config.model == "ChromTransfer-Reg":
        model = ChromTransfer_Reg(cis_feature_num=sum(config.cobindingTFs_mask))
        
    model.to(device)
    model.load_state_dict(torch.load(f"best_model.pth"))
    model.eval()
    test_loss = 0.0
    total_accuracy = 0.0
    all_labels = []
    all_predictions = []
    all_region_nums = []
    criterion = nn.BCELoss()
    with torch.no_grad():
        for d in tqdm(test_loader):
            one_hot_seqs = d['one_hot_seq'].float().to(device)
            labels_classifier = d['label_classifier'].float().to(device)
            datas_FUNCODE = d['data_FUNCODE'].float().to(device)
            cobindingTFs_data = d['cobindingTFs'].float().to(device)
            region_nums = d['region_num']
            outputs_classifier = model(one_hot_seqs, datas_FUNCODE, cobindingTFs_data)
            loss = criterion(outputs_classifier, labels_classifier.unsqueeze(1))
            accuracy = calculate_accuracy(outputs_classifier, labels_classifier)
            total_accuracy += accuracy
            test_loss += loss.item() * one_hot_seqs.size(0)
            all_labels.extend(labels_classifier.view(-1).cpu().numpy())
            all_predictions.extend(outputs_classifier.view(-1).cpu().numpy())
            all_region_nums.extend(region_nums)

    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = total_accuracy / len(test_loader.dataset)

    roc_auc = roc_auc_score(all_labels, all_predictions)
    binary_predictions = [1 if x >= 0.5 else 0 for x in all_predictions]
    precision = precision_score(all_labels, binary_predictions)
    recall = recall_score(all_labels, binary_predictions)
    mcc = matthews_corrcoef(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)

    precision2, recal2, _ = precision_recall_curve(all_labels, all_predictions)
    auprc = auc(recal2, precision2)

    all_predictions = np.array(all_predictions)
    all_region_nums = np.array(all_region_nums)
    all_predictions_df = pd.DataFrame({'region_num': all_region_nums, 'prediction': all_predictions})

    return test_loss, test_accuracy, roc_auc, precision, recall, mcc, f1, auprc, all_predictions_df


def train_model(val_loader):
    best_valid_auprc = 0
    best_epoch = 0

    if config.model == "ChromTransfer-Base":
        model = ChromTransfer_Base()
    elif config.model == "ChromTransfer-Cons":
        model = ChromTransfer_Cons()
    elif config.model == "ChromTransfer-Reg":
        model = ChromTransfer_Reg(cis_feature_num=sum(config.cobindingTFs_mask))
        
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    reach_max_epoch = False
    for epoch in range(config.num_epochs):
        epoch_source = epoch
        file_path_source = f"{config.source_species}.chrOthers_epoch{epoch_source + 1}.txt"
        if reach_max_epoch == False:
            max_epoch = epoch_source
        if not os.path.exists(file_path_source):
            reach_max_epoch = True
            file_path_source = f"{config.source_species}.chrOthers_epoch{(epoch_source - max_epoch) % max_epoch + 1}.txt"

        train_dataset = SupervisedDataset(file_path_source, config.FUNCODE_file[config.source_species], config.DNA_file[config.source_species], config.cobindingTFs_file[config.source_species], config.cobindingTFs_mask)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

        model.train()
        train_loss = 0.0
        total_accuracy = 0.0

        # training
        for d in tqdm(train_loader):
            one_hot_seqs = d['one_hot_seq'].float().to(device)
            labels_classifier = d['label_classifier'].float().to(device)
            datas_FUNCODE = d['data_FUNCODE'].float().to(device)
            cobindingTFs_data = d['cobindingTFs'].float().to(device)
            optimizer.zero_grad()
            outputs_classifier = model(one_hot_seqs, datas_FUNCODE, cobindingTFs_data)
            loss = criterion(outputs_classifier, labels_classifier.unsqueeze(1))

            accuracy = calculate_accuracy(outputs_classifier, labels_classifier)
            total_accuracy += accuracy
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * one_hot_seqs.size(0)

        # log training loss
        with open(f"train_loss.txt", 'a') as log_file:
            train_loss = train_loss / len(train_loader.dataset)
            train_accuracy = total_accuracy / len(train_loader.dataset)
            log_file.write(f'Epoch: {epoch+1}\tTraining Loss: {train_loss:.4f}\tTraining Accuracy: {train_accuracy:.4f}\n')
            
        print(f'Epoch: {epoch+1}\tTraining Loss: {train_loss:.4f}\tTraining Accuracy: {train_accuracy:.4f}\n')
        
        # validation
        valid_loss, valid_accuracy, valid_auc, valid_precision, valid_recall, valid_mcc, valid_f1, valid_auprc = validate_model(model, val_loader)
        with open(f"valid_loss_{config.source_species}.txt", 'a') as log_file:
            log_file.write(f'Epoch: {epoch+1}\tValidation Loss: {valid_loss:.4f}\tValidation Accuracy: {valid_accuracy:.4f}\tValidation AUC: {valid_auc:.4f}\tValidation Precision: {valid_precision}\tValidation Recall: {valid_recall:.4f}\tValidation MCC: {valid_mcc:.4f}\tValidation F1: {valid_f1:.4f}\tValidation auprc: {valid_auprc:.4f}\n')
            
        print(f'Epoch: {epoch+1}\tValidation Loss: {valid_loss:.4f}\tValidation Accuracy: {valid_accuracy:.4f}\tValidation AUC: {valid_auc:.4f}\tValidation Precision: {valid_precision}\tValidation Recall: {valid_recall:.4f}\tValidation MCC: {valid_mcc:.4f}\tValidation F1: {valid_f1:.4f}\tValidation auprc: {valid_auprc:.4f}\n')

        # save best model
        if valid_auprc > best_valid_auprc:
            best_valid_auprc = valid_auprc
            best_epoch = epoch
            torch.save(model.state_dict(), f"best_model.pth")
            print(f'best Epoch: {best_epoch+1}\tbest auprc: {best_valid_auprc:.4f}\n')

        # early stopping
        if epoch - best_epoch > 5:
            break

    return


if __name__ == "__main__":
    print(f"start training for {config.tf}")

    # validation dataset
    val_dataset = SupervisedDataset(
        f"{config.source_species}.chr1_random100000.txt",
        config.FUNCODE_file[config.source_species],
        config.DNA_file[config.source_species],
        config.cobindingTFs_file[config.source_species],
        config.cobindingTFs_mask
    )
    val_loader = DataLoader(val_dataset, batch_size=config.predict_batch_size, shuffle=False, num_workers=config.predict_num_workers)

    # training and validation
    train_model(val_loader=val_loader)

    # testing
    for genome in [config.target_species, config.source_species]:
        if (genome == config.target_species) and (config.peak_file_target == ""):
            print(f"No peak file in target species, skipping testing on {genome}")
            continue
        
        # testing dataset
        test_dataset = SupervisedDataset(
            f"{genome}.chr2.txt",
            config.FUNCODE_file[genome],
            config.DNA_file[genome],
            config.cobindingTFs_file[genome],
            config.cobindingTFs_mask
        )
        test_loader = DataLoader(test_dataset, batch_size=config.predict_batch_size, num_workers=config.predict_num_workers)
        test_loss, test_accuracy, test_roc_auc, test_precision, test_recall, test_mcc, test_f1, test_auprc, test_predictions_df = test_model(test_loader=test_loader)
        test_predictions_df.to_csv(f"predictions_{genome}Chr2.txt", index=False)
        
        # log test results
        with open(f"test_loss_{genome}.txt", 'w') as log_file:
            log_file.write(f'Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\nTest AUC: {test_roc_auc:.4f}\nTest Precision: {test_precision}\nTest Recall: {test_recall:.4f}\nTest MCC: {test_mcc:.4f}\nTest F1: {test_f1:.4f}\nTest auprc: {test_auprc:.4f}\n')
        
        # print test results
        print(f'Test Loss: {test_loss:.4f}\nTest Accuracy: {test_accuracy:.4f}\nTest AUC: {test_roc_auc:.4f}\nTest Precision: {test_precision}\nTest Recall: {test_recall:.4f}\nTest MCC: {test_mcc:.4f}\nTest F1: {test_f1:.4f}\nTest auprc: {test_auprc:.4f}\n')
