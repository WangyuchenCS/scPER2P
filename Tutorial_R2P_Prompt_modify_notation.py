import copy
import datetime
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
import loralib as lora
import pickle
import torch
import scanpy as sc
import numpy as np
import wandb
from scipy.sparse import issparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)

sys.path.insert(0, "../")
import scgpt_PEFT as scg
from scgpt_PEFT.model.model_prompt import TransformerModel, AdversarialDiscriminator

from scgpt_PEFT.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt_PEFT.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)
from scgpt_PEFT.tokenizer.gene_tokenizer import GeneVocab
from scgpt_PEFT.preprocess import Preprocessor
from scgpt_PEFT import SubsetsBatchSampler
from scgpt_PEFT.utils import set_seed, category_str2int, eval_scib_metrics
from sklearn import preprocessing


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine



def calculate_metrics(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError("The arrays must have the same shape.")

    def compute_metrics(a1, a2):
        mse = mean_squared_error(a1, a2)
        mae = np.mean(np.abs(a1 - a2))
        pcc_values = [pearsonr(a1[i], a2[i])[0] for i in range(a1.shape[0])]
        avg_pcc = np.nanmean(pcc_values)  
        r2_values = [r2_score(a1[i], a2[i]) for i in range(a1.shape[0])]
        avg_r2 = np.mean(r2_values)
        cosine_sim_values = [1 - cosine(a1[i], a2[i]) for i in range(a1.shape[0])]
        avg_cosine_sim = np.mean(cosine_sim_values)

        return mse, mae, avg_pcc, avg_r2, avg_cosine_sim

    sample_metrics = compute_metrics(array1, array2)
    feature_metrics = compute_metrics(array1.T, array2.T)
    
    return sample_metrics, feature_metrics


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
        data_pt: Dict[str, torch.Tensor],
        batch_size: int,
        shuffle: bool = False,
        intra_domain_shuffle: bool = False,
        drop_last: bool = False,
        num_workers: int = 0,
        sampler:str=None,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)



    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    return data_loader


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total:{}'.format(total_num))
    print('trainable:{}'.format(trainable_num))


def train(model: nn.Module, loader: DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    (   total_loss,
        total_mse,
        total_cls,
        total_cce,
        total_mvc,
        total_ecs,
        total_dab,
        total_adv_E,
        total_adv_D,
        total_zero_log_prob,
        total_mvc_zero_log_prob,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    total_error = 0.0
    start_time = time.time()

    num_batches = len(loader)
    for batch, batch_data in enumerate(loader):
        input_gene_ids = batch_data["gene_ids"].to(device)
        input_values = batch_data["values"].to(device)
        target_values = batch_data["target_values"].float().to(device)
        batch_labels = batch_data["batch_labels"].to(device)
        celltype_labels = batch_data["celltype_labels"].to(device)

        src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])  
        if use_prompt and prompt_type == 'prefix_prompt':
            bool_tensor = torch.empty((input_gene_ids.shape[0],num_tokens), dtype=torch.bool).to(device)
            bool_tensor.fill_(src_key_padding_mask[0][0].item()).to(device)
            src_key_padding_mask = torch.cat((bool_tensor,src_key_padding_mask),dim=1)


        with torch.cuda.amp.autocast(enabled=config.amp): 
            output_dict = model(input_gene_ids, input_values,  
                                src_key_padding_mask=src_key_padding_mask,
                                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                                CLS=CLS, CCE=CCE, MVC=MVC, ECS=ECS, do_sample=do_sample_in_train,  # False
                            )

            masked_positions = input_values.eq(mask_value) 
            loss = 0.0
            metrics_to_log = {}
            if MLM:
                loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)  
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}


        model.zero_grad() 
        scaler.scale(loss).backward() 
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False if scaler.is_enabled() else True,)  

            if len(w) > 0:
                logger.warning(f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler." )
        scaler.step(optimizer)
        scaler.update()


        wandb.log(metrics_to_log)

        total_loss += loss.item()
        total_mse += loss_mse.item() if MLM else 0.0
        total_zero_log_prob += loss_zero_log_prob.item() if explicit_zero_prob else 0.0
        total_mvc_zero_log_prob += (loss_mvc_zero_log_prob.item() if MVC and explicit_zero_prob else 0.0)
        # total_error += error_rate
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            cur_cls = total_cls / log_interval if CLS else 0.0
            cur_cce = total_cce / log_interval if CCE else 0.0
            cur_mvc = total_mvc / log_interval if MVC else 0.0
            cur_ecs = total_ecs / log_interval if ECS else 0.0
            cur_dab = total_dab / log_interval if DAB else 0.0

            cur_zero_log_prob = (
                total_zero_log_prob / log_interval if explicit_zero_prob else 0.0
            )
            cur_mvc_zero_log_prob = (
                total_mvc_zero_log_prob / log_interval
                if MVC and explicit_zero_prob
                else 0.0
            )
            cur_error = total_error / log_interval
            # ppl = math.exp(cur_loss)
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | "
                + (f"mse {cur_mse:5.2f} | mre {cur_error:5.2f} |" if MLM else "")
                + (f"cls {cur_cls:5.2f} | " if CLS else "")
                + (f"err {cur_error:5.2f} | " if CLS else "")
                + (f"cce {cur_cce:5.2f} |" if CCE else "")
                + (f"mvc {cur_mvc:5.2f} |" if MVC else "")
                + (f"ecs {cur_ecs:5.2f} |" if ECS else "")
                + (f"dab {cur_dab:5.2f} |" if DAB else "")
                # + (f"adv_E {cur_adv_E:5.2f} |" if ADV else "")
                # + (f"adv_D {cur_adv_D:5.2f} |" if ADV else "")
                + (f"nzlp {cur_zero_log_prob:5.2f} |" if explicit_zero_prob else "")
                + (
                    f"mvc_nzlp {cur_mvc_zero_log_prob:5.2f} |"
                    if MVC and explicit_zero_prob
                    else ""
                )
            )
            total_loss = 0
            total_mse = 0
            total_cls = 0
            total_cce = 0
            total_mvc = 0
            total_ecs = 0
            total_dab = 0
            total_zero_log_prob = 0
            total_mvc_zero_log_prob = 0
            total_error = 0
            start_time = time.time()


def define_wandb_metrcis():
    wandb.define_metric("valid/mse", summary="min", step_metric="epoch")
    wandb.define_metric("valid/mre", summary="min", step_metric="epoch")
    wandb.define_metric("valid/dab", summary="min", step_metric="epoch")
    wandb.define_metric("valid/sum_mse_dab", summary="min", step_metric="epoch")
    wandb.define_metric("test/avg_bio", summary="max")


def evaluate(model: nn.Module, loader: DataLoader, return_raw: bool = False) -> float:
    """
    Evaluate the model on the evaluation data.
    """
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_dab = 0.0
    total_num = 0
    predictions = []
    output_dict_ = []
    batch_data_ = []
    output_dict_storage = dict()
    batch_data_storage = dict()
    with torch.no_grad():
        for batch_data in loader:
            input_gene_ids = batch_data["gene_ids"].to(device)
            input_values = batch_data["values"].to(device)
            target_values = batch_data["target_values"].to(device)
            batch_labels = batch_data["batch_labels"].to(device)
            celltype_labels = batch_data["celltype_labels"].to(device)

            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            if use_prompt and prompt_type == 'prefix_prompt':
                bool_tensor = torch.empty((input_gene_ids.shape[0], num_tokens), dtype=torch.bool).to(device)
                bool_tensor.fill_(src_key_padding_mask[0][0].item()).to(device)
                src_key_padding_mask = torch.cat((bool_tensor, src_key_padding_mask), dim=1)
            with torch.cuda.amp.autocast(enabled=config.amp):
                output_dict = model(  
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                    CLS=CLS,  # evaluation does not need CLS or CCE
                    CCE=False,
                    MVC=False,
                    ECS=False,
                    do_sample=do_sample_in_train,

                )
                masked_positions = input_values.eq(mask_value)  # the postions to predict
                loss = 0.0
                output_values = output_dict["mlm_output"]
                loss_mse = criterion(output_dict["mlm_output"], target_values, masked_positions)
                loss = loss + loss_mse
                metrics_to_log = {"train/mse": loss_mse.item()}

            total_loss += loss.item() * len(input_gene_ids)
            total_num += len(input_gene_ids)
            preds = output_values.cpu().numpy()
            predictions.append(preds)

            output_dict_.append(output_dict)
            batch_data_.append(batch_data)
            

    for od in output_dict_:
        for key, value in od.items():
            if key not in output_dict_storage:
                output_dict_storage[key] = []
            output_dict_storage[key].append(value.detach().cpu().numpy())
    for key in output_dict_storage:
        output_dict_storage[key] = np.concatenate(output_dict_storage[key], axis=0)


    for bd in batch_data_:
        for key, value in bd.items():
            if key not in batch_data_storage:
                batch_data_storage[key] = []
            batch_data_storage[key].append(value.numpy())
    for key in batch_data_storage:
        batch_data_storage[key] = np.concatenate(batch_data_storage[key], axis=0) # torch.cat(batch_data_storage[key], dim=0)



    wandb.log(
        {
            "valid/mse": total_loss / total_num,
            "valid/err": total_error / total_num,
            "valid/dab": total_dab / total_num,
            "valid/sum_mse_dab": (total_loss + dab_weight * total_dab) / total_num,
            "epoch": epoch,
        },
    )

    if return_raw:
        return np.concatenate(predictions, axis=0)
    return total_loss / total_num, output_dict_storage, batch_data_storage  # Target_dict


class EarlyStopping():
    def __init__(self, patience=20, min_delta=0.000001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss, model):
        if self.best_loss != None:
            print(f"best_loss: {self.best_loss}, min_delta {self.min_delta}, val_loss {val_loss}")
            print(f"Loss error: {self.best_loss - val_loss}")
        if self.best_loss == None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            logger.info(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                logger.info('INFO: Early stopping')
                self.early_stop = True

    def save_checkpoint(self, model):
        '''Saves model when validation loss decreases.'''
        self.save_path = "../"+args.dataset+"/"+args.prompt_type+"/"+str(args.lr)+"/"+str(args.seed_split)+"/"+str(args.scale)+"/"+'best_model.pt'
        parent_dir = os.path.dirname(self.save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
            print(f"Directory {parent_dir} created")
        else:
            print(f"Directory {parent_dir} already exists")

        torch.save(model.state_dict(), self.save_path)

































# if __name__ == "__main__":

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='dataset3',help='dataset1/dataset2/dataset3')
parser.add_argument("--lr", type=float, default=3e-4, help='Learning rate.')

parser.add_argument("--use_prompt", type=bool, default=True, help='whether use prompt or not.')
parser.add_argument("--prompt_type", type=str, default='Gene_encoder_prompt',help=' " ", prefix_prompt, Gene_token_prompt, Gene_encoder_prompt, LoRA')

parser.add_argument("--data_name", type=str, default='ms',help='ms')
parser.add_argument("--data_path", type=str, default='../data/', help='Path of data for predicting.')
parser.add_argument("--space_conf", type=str, default=[1,1,1,1,1,1,0,0,0,0,0,0],help='encoder space adapter list')
parser.add_argument("--mlp_conf", type=str, default=[1,1,1,1,1,1,0,0,0,0,0,0],help='encoder mlp adapter list')
parser.add_argument("--epoch", type=int, default=80, help='Number of epochs.')
parser.add_argument("--scale", type=float, default=0.0, help='scale of residual adapter.')
parser.add_argument("--flag", type=str, default="All encoder adapter",help=' "All encoder adapter", "space_adapter only", "mlp_adapter only"')
parser.add_argument("--seed_split", type=int, default=42, help='seed_split.')


args = parser.parse_args()


print(args) 



hyperparameter_defaults = dict(
    seed=42,
    dataset_name=args.data_name,
    do_train=True,
    load_model="../scGPT_human",
    mask_ratio=0.0,
    epochs=args.epoch,
    n_bins=51,
    n_hvg = 1200 if args.dataset == 'dataset1' else False,
    n_hvp = 4000,
    MVC=False, 
    ecs_thres=0.0, 
    dab_weight=0.0,
    lr=args.lr,
    batch_size=6,
    layer_size=128,
    nlayers=4,  
    nhead=4, 
    dropout=0.2, 
    schedule_ratio=0.9, 
    save_eval_interval=5,
    fast_transformer= False,  
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    include_zero_gene = True, # False,
    freeze = False, #freeze
    DSBN = False,  # Domain-spec batchnorm
    data_path=args.data_path,
    use_prompt=args.use_prompt,
    prompt_type=args.prompt_type, 
    num_tokens=64,
    n_layers_conf=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # token
    mlp_adapter_conf=args.space_conf,
    space_adapter_conf=args.mlp_conf,
    scale = args.scale,
    flag = "space_adapter only" if args.scale < 0.2 else args.flag,
    seed_split = args.seed_split,
)



#%%
run = wandb.init(
    config=hyperparameter_defaults,
    project="scGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)
set_seed(config.seed)

pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"
include_zero_gene = config.include_zero_gene 
max_seq_len = 2001
n_bins = config.n_bins

input_style = "binned"  
output_style = "log1p" 

MLM = True 
CLS = False  
ADV = False  
CCE = False  
MVC = config.MVC  
ECS = config.ecs_thres > 0  
DAB = False  
INPUT_BATCH_LABELS = False  
input_emb_style = "continuous" 
cell_emb_style = "cls" 
adv_E_delay_epochs = 0  
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres  # 0
dab_weight = config.dab_weight
explicit_zero_prob = False and MLM and include_zero_gene 
do_sample_in_train = False and explicit_zero_prob  
per_seq_batch_sample = False

lr = config.lr 
lr_ADV = 1e-5 
batch_size = config.batch_size
eval_batch_size = config.batch_size
epochs = config.epochs
schedule_interval = 1

fast_transformer = config.fast_transformer
fast_transformer_backend = "flash" 
embsize = config.layer_size  
d_hid = config.layer_size  
nlayers = config.nlayers 
nhead = config.nhead 
dropout = config.dropout  
data_path = config.data_path
use_prompt = config.use_prompt
prompt_type = config.prompt_type
num_tokens = config.num_tokens
n_layers_conf = config.n_layers_conf
mlp_adapter_conf = config.mlp_adapter_conf
space_adapter_conf = config.space_adapter_conf

log_interval = 100  # iterations
save_eval_interval = config.save_eval_interval  # epochs
do_eval_scib_metrics = True

# %% validate settings
assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins


dataset_type = args.dataset
if prompt_type == " ":
    prompt_type_name = "full_finetune"
else:
    prompt_type_name = prompt_type 
save_dir = Path(f"../{dataset_type}/{prompt_type_name}/{lr}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")
logger = scg.logger
scg.utils.add_file_handler(logger, save_dir / "run.log")

current_time = datetime.datetime.now()
logger.info(f"{current_time}")
print(f"Current time: {current_time}")


def load_dataset(dataset):

    mapping = pd.read_csv('../mapping_0203.csv')
    if dataset == 'dataset1':
        adata_gene = sc.read("../dataset/dataset1.h5ad")  # [:1600]
        adata_protein = sc.read("../dataset/dataset1_p.h5ad")  # [:1600]

    elif dataset == 'dataset2':
        adata_gene = sc.read("../dataset/dataset2.h5ad")
        adata_protein = sc.read("../dataset/dataset2_p.h5ad")
    elif dataset == 'dataset3':
        adata_gene = sc.read("../dataset/dataset3.h5ad")
        adata_protein = sc.read("../dataset/dataset3_p.h5ad")

    mask = adata_protein.var['Entrez_Gene_Id'].isin(mapping['initial_alias'].astype(str).values)
    filtered_adata_protein = adata_protein[:, mask]
    adata_protein = filtered_adata_protein.copy()
    print("adata_gene", adata_gene.shape)
    print("adata_protein", adata_protein.shape)
    return adata_gene, adata_protein

dataset = args.dataset

adata, adata_protein = load_dataset(dataset)
adata_protein.var.index = ['p_' + i for i in adata_protein.var.index] 




le = preprocessing.LabelEncoder()
if dataset == 'dataset1':
    encoded_batch = 0
    adata.obs["celltype"] = 0 
    celltype_id_labels = adata.obs["celltype.l1"].astype("category").cat.codes.values 
    adata.var["gene_name"] = adata.var.index.tolist()


elif dataset == 'dataset2':
    encoded_batch = le.fit_transform([i.split('-')[-1] for i in adata.obs['cell'].values]) 
    adata.obs["celltype"] = 0
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values 
    adata.var["gene_name"] = adata.var["Gene symbol"].tolist()


elif dataset == 'dataset3':
    encoded_batch = 0 
    adata.obs["celltype"] = 0
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values 
    adata.var["gene_name"] = adata.var["Gene symbol"].tolist()

adata.obs["batch_id"] =  encoded_batch
adata.obs["str_batch"] = adata.obs["batch_id"].astype('category')
data_is_raw = False



num_types = len(np.unique(celltype_id_labels))
print(f'celltype num_types:{num_types}')
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels


config.use_mod = True
if config.use_mod:
    gene_rna_df = pd.DataFrame(index = adata.var.index.tolist())
    gene_rna_df['mod'] = 'RNA'
    gene_protein_df = pd.DataFrame(index = adata_protein.var.index.tolist())
    gene_protein_df['mod'] = 'Protein'
    gene_loc_df = pd.concat([gene_rna_df, gene_protein_df])
    gene_loc_df['mod'] = gene_loc_df['mod'].astype('category')

# =========================================================================================================== 






if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    old_vocab = vocab

    adata.var["id_in_vocab"] = [1 if gene in vocab else -1 for gene in adata.var["gene_name"] ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info( f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes " f"in vocabulary of size {len(vocab)}."   )
    adata = adata[:, adata.var["id_in_vocab"] >= 0] 

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(f"Resume model from {model_file}, the model args will override the "  f"config {model_config_file}."  )
    embsize = model_configs["embsize"]  # 5
    nhead = model_configs["nheads"]  # 
    d_hid = model_configs["d_hid"]  # 5
    nlayers = model_configs["nlayers"]  # 1
    n_layers_cls = model_configs["n_layers_cls"]  # 3


preprocessor = Preprocessor(
    use_key="X", 
    filter_gene_by_counts=1, 
    filter_cell_by_counts=1, 
    normalize_total=1e4, 
    result_normed_key="X_normed", 
    log1p=data_is_raw, 
    result_log1p_key="X_log1p",
    subset_hvg=config.n_hvg, 
    hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
    binning=config.n_bins, 
    result_binned_key="X_binned", 
)
preprocessor(adata, batch_key=None)

protein = copy.deepcopy(adata_protein)

preprocessor_protein = Preprocessor(
    use_key="X", 
    filter_gene_by_counts=0,  
    filter_cell_by_counts=False, 
    normalize_total=True,
    result_normed_key="X_normed", 
    log1p=True, 
    result_log1p_key="X_log1p",
    subset_hvg=False,  
    hvg_flavor=None,
    binning=None, 
    result_binned_key="X_binned",  
)

input_layer_key = { "normed_raw": "X_normed",
                    "log1p": "X_normed",
                    "binned": "X_binned",
                  }[input_style] 
all_counts = ( adata.layers[input_layer_key].A if issparse(adata.layers[input_layer_key]) else adata.layers[input_layer_key])



def prot_log1p(protein):
    sc.pp.log1p(protein)
    return protein
protein = prot_log1p(protein)
all_protein_counts = protein.X





adata.var["gene_name"] = adata.var.index.tolist()
genes = adata.var["gene_name"].tolist()


celltypes_labels = adata.obs["celltype_id"].tolist() 
num_types = len(set(celltypes_labels))
celltypes_labels = np.array(celltypes_labels)


batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)



if config.use_mod:
    mod_type = np.array([gene_loc_df.loc[g, 'mod'] for g in genes])
    vocab_mod = Vocab(VocabPybind(np.unique(gene_loc_df['mod']).tolist() + special_tokens, None))
    vocab_mod.set_default_index(vocab_mod["<pad>"])
    mod_type = np.array(vocab_mod(list(mod_type)), dtype=int)
    ntokens_mod = len(vocab_mod)



seed_g = config.seed_split

indices = np.arange(adata.n_obs)


(   train_data,
    valid_data,
    train_protein_data,
    valid_protein_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split( all_counts, all_protein_counts, celltypes_labels, batch_ids, test_size=0.1, shuffle=True, random_state=seed_g)  # , random_state=42



if config.load_model is None:
    vocab = Vocab(VocabPybind(genes + special_tokens, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)
else:
    pretrained_genes = [g for g in genes + special_tokens if g in old_vocab] 
    new_genes = [g for g in genes + special_tokens if g not in old_vocab]
    gene_ids_pretrained = np.array(old_vocab(pretrained_genes), dtype=int)

    vocab = Vocab(VocabPybind(pretrained_genes + new_genes, None))
    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(vocab(genes), dtype=int)





tokenized_train = tokenize_and_pad_batch(  
    train_data,  # 
    gene_ids,  # 1
    max_len=max_seq_len,  # 2
    vocab=vocab,
    pad_token=pad_token,  # '
    pad_value=pad_value,  # -
    append_cls=True,  #
    include_zero_gene=include_zero_gene,
)  
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)


logger.info(f"train set number of samples: {tokenized_train['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_train['genes'].shape[1]}")
logger.info(f"valid set number of samples: {tokenized_valid['genes'].shape[0]}, "
            f"\n\t feature length: {tokenized_valid['genes'].shape[1]}")










device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(vocab)  

model = TransformerModel(
    ntokens, 
    embsize,  #
    nhead,  # 
    d_hid,  # 5
    nlayers,  # 
    nlayers_cls=3,  # 
    n_cls=num_types if CLS else 1, #
    vocab=vocab,  # 
    dropout=dropout,  # 
    pad_token=pad_token,  # '
    pad_value=pad_value,  # 
    do_mvc=MVC,  
    do_dab=DAB,  
    use_batch_labels=INPUT_BATCH_LABELS,  
    num_batch_labels=num_batch_types, 
    domain_spec_batchnorm=config.DSBN, 
    input_emb_style=input_emb_style, 
    n_input_bins=n_input_bins, 
    cell_emb_style=cell_emb_style, 
    mvc_decoder_style=mvc_decoder_style, 
    ecs_threshold=ecs_threshold,  # 0.0
    explicit_zero_prob=explicit_zero_prob, 
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend, 
    pre_norm=config.pre_norm,  
    batch_size=batch_size,  # 1
    use_prompt=use_prompt, 
    num_tokens=num_tokens,  
    scale=config.scale,  
    flag = config.flag,
    prompt_type=prompt_type, 
    n_layers_conf=n_layers_conf,
    mlp_adapter_conf=mlp_adapter_conf, 
    space_adapter_conf=space_adapter_conf, 
    max_len=max_seq_len, # 
    input_shape = tokenized_train['values'].shape[1],  # 
    output_shape = train_protein_data.shape[1],  
)
if config.load_model is not None: 
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        print('except!!!! only load params that are in the model and match the size!!!!!')
        use_flash_attn = getattr(model, "use_fast_transformer", True)  
        pretrained_dict = torch.load(model_file)
        if not use_flash_attn and prompt_type != "LoRA":
            pretrained_dict = {k.replace("Wqkv.", "in_proj_"): v for k, v in pretrained_dict.items()}

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                        if k in model_dict and v.shape == model_dict[k].shape }

        for k, v in pretrained_dict.items():
            logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)


pre_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() ).values())
print(f"Total Pre freeze Params {(pre_freeze_param_count )}")
# 51537427


if use_prompt and prompt_type in ["prefix_prompt", "Gene_encoder_prompt", "Gene_token_prompt", "LoRA"]:
    print('All para.requires_grad:  False, Freeze!')
    for name, para in model.named_parameters():
        para.requires_grad = False # False 
else:
    print('All para.requires_grad:  True  Trainable!')
    for name, para in model.named_parameters():
        para.requires_grad = True # False 


for param in model.decoder.parameters():
    param.requires_grad = True


for name, para in model.named_parameters():
    if 'lora_' in name:
        print('lora_ in name:', name)
        logger.info(f"lora_ in name: {name}")
        para.requires_grad = True
    if 'cls' in name:
        print('cls in name:', name)
        logger.info(f"cls in name: {name}")
        para.requires_grad = True
    if 'prompt_embeddings' in name:
        print('prompt_embeddings in name:', name)
        logger.info(f"prompt_embeddings in name: {name}")
        para.requires_grad = True
    if 'Adapter' in name:
        print('Adapter in name:', name)
        logger.info(f"Adapter in name: {name}")
        para.requires_grad = True
post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad ).values())
print(f"Total Post0 freeze Params {(post_freeze_param_count )}")
# 733202

if use_prompt and prompt_type == "LoRA":
    def lora_bias_trainable(model: nn.Module, bias: str = 'none') -> None:
        if bias == 'none':
            return
        elif bias == 'all':
            for n, p in model.named_parameters():
                if 'bias' in n:
                    p.requires_grad = True
        elif bias == 'lora_only':
            for m in model.modules():

                if isinstance(m, lora.LoRALayer) and hasattr(m, 'bias') and m.bias is not None:
                    m.bias.requires_grad = True
        else:
            raise NotImplementedError
    lora_bias_trainable(model, bias='lora_only')

get_parameter_number(model)  # for print
post_freeze_param_count = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters() if p.requires_grad ).values())
print(f"Total Post freeze Params {(post_freeze_param_count )}")


logger.info(f"Total Pre freeze Params {(pre_freeze_param_count )}")
logger.info(f"Total Post freeze Params {(post_freeze_param_count )}")
wandb.log({ "info/pre_freeze_param_count": pre_freeze_param_count,
            "info/post_freeze_param_count": post_freeze_param_count,},)
model.to(device)
wandb.watch(model)

criterion = masked_mse_loss 
optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-4 if config.amp else 1e-8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=config.schedule_ratio)
scaler = torch.cuda.amp.GradScaler(enabled=config.amp) 

best_val_loss = float("inf")
best_avg_bio = 0.0
best_model = None
define_wandb_metrcis()





def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )

    input_gene_ids_train, input_gene_ids_valid = ( tokenized_train["genes"], tokenized_valid["genes"],)

    input_values_train, input_values_valid = masked_values_train, masked_values_valid  

 
    target_values_train, target_values_valid = (train_protein_data, valid_protein_data,)

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()  #
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()
    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()  
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train, 
        "target_values": target_values_train, 
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid, 
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
    }
    return train_data_pt, valid_data_pt





train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)  # False


train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=batch_size,
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
    # sampler=train_sampler,
)
valid_loader = prepare_dataloader(
    valid_data_pt,
    batch_size=eval_batch_size,
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
)





early_stopping = EarlyStopping()
for epoch in range(1, epochs + 1):
    print("Epoch: ", epoch)
    epoch_start_time = time.time()

    if config.do_train:
        print("Training model")
        train(model, loader=train_loader,)

    print("Evaluating model")
    val_loss, output_dict, Target_dict = evaluate(model, loader=valid_loader,)

    elapsed = time.time() - epoch_start_time
    logger.info("-" * 89)
    logger.info( f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | " f"valid loss/mse {val_loss:5.4f} |"  )
    logger.info("-" * 89)



    scheduler.step()

    early_stopping(val_loss,model) # 
    if early_stopping.early_stop or epoch == epochs:
        if early_stopping.early_stop:
            print("Early stopping with epoch:{:d} !!".format(epoch))
            logger.info("Early stopping with epoch:{:d} !!".format(epoch))
        else:
            print("Training complete with epoch:{:d} !!".format(epoch))
            logger.info("Training complete with epoch:{:d} !!".format(epoch))

        model_path = "../"+args.dataset+"/"+args.prompt_type+"/"+str(args.lr)+"/"+str(args.seed_split)+"/"+str(args.scale)+"/"+'best_model.pt'
                            
        model.load_state_dict(torch.load(model_path))

        val_loss, output_dict, Target_dict = evaluate(model, loader=valid_loader,)
        # break

        all_sample_mse = []
        all_sample_mae = []
        all_sample_r2 = []
        all_sample_pcc = []
        all_sample_cosine_similarity = []

        all_feature_mse = []
        all_feature_mae = []
        all_feature_r2 = []
        all_feature_pcc = []
        all_feature_cosine_similarity = []

        real_y = Target_dict['target_values']
        pred_y = output_dict['mlm_output']


        sample_metrics, feature_metrics = calculate_metrics(real_y, pred_y)

        all_sample_mse.append(sample_metrics[0])
        all_sample_mae.append(sample_metrics[1])
        all_sample_r2.append(sample_metrics[3])
        all_sample_pcc.append(sample_metrics[2])
        all_sample_cosine_similarity.append(sample_metrics[4])

        all_feature_mse.append(feature_metrics[0])
        all_feature_mae.append(feature_metrics[1])
        all_feature_r2.append(feature_metrics[3])
        all_feature_pcc.append(feature_metrics[2])
        all_feature_cosine_similarity.append(feature_metrics[4])

        print('By feature: ', 'MSE: ', str(np.mean(all_sample_mse).round(4)), 'MAE: ', str(np.mean(all_sample_mae).round(4)), 'R2: ', str(np.mean(all_sample_r2).round(4)), 'PCC: ', str(np.mean(all_sample_pcc).round(4)), 'Cosine Similarity: ', str(np.mean(all_sample_cosine_similarity).round(4)))
        print('By sample: ', 'MSE: ', str(np.mean(all_feature_mse).round(4)), 'MAE: ', str(np.mean(all_feature_mae).round(4)), 'R2: ', str(np.mean(all_feature_r2).round(4)), 'PCC: ', str(np.mean(all_feature_pcc).round(4)), 'Cosine Similarity: ', str(np.mean(all_feature_cosine_similarity).round(4)))
        logger.info(f"By feature: MSE: {np.mean(all_sample_mse).round(4)}±{np.std(all_sample_mse).round(4)} MAE: {np.mean(all_sample_mae).round(4)}±{np.std(all_sample_mae).round(4)} R2: {np.mean(all_sample_r2).round(4)}±{np.std(all_sample_r2).round(4)} PCC: {np.mean(all_sample_pcc).round(4)}±{np.std(all_sample_pcc).round(4)} Cosine Similarity: {np.mean(all_sample_cosine_similarity).round(4)}±{np.std(all_sample_cosine_similarity).round(4)}")
        logger.info(f"By sample: MSE: {np.mean(all_feature_mse).round(4)}±{np.std(all_feature_mse).round(4)} MAE: {np.mean(all_feature_mae).round(4)}±{np.std(all_feature_mae).round(4)} R2: {np.mean(all_feature_r2).round(4)}±{np.std(all_feature_r2).round(4)} PCC: {np.mean(all_feature_pcc).round(4)}±{np.std(all_feature_pcc).round(4)} Cosine Similarity: {np.mean(all_feature_cosine_similarity).round(4)}±{np.std(all_feature_cosine_similarity).round(4)}")
        print(config)


    real_y = Target_dict['target_values']
    pred_y = output_dict['mlm_output']

    sample_metrics, feature_metrics = calculate_metrics(real_y, pred_y)
    all_sample_mse = sample_metrics[0]
    all_sample_mae = sample_metrics[1]
    all_sample_r2 = sample_metrics[3]
    all_sample_pcc = sample_metrics[2]
    all_sample_cosine_similarity = sample_metrics[4]

    all_feature_mse = feature_metrics[0]
    all_feature_mae = feature_metrics[1]
    all_feature_r2 = feature_metrics[3]
    all_feature_pcc = feature_metrics[2]
    all_feature_cosine_similarity = feature_metrics[4]

    print('By feature: ', 'MSE: ', str(np.mean(all_sample_mse).round(4)), 'MAE: ', str(np.mean(all_sample_mae).round(4)), 'R2: ', str(np.mean(all_sample_r2).round(4)), 'PCC: ', str(np.mean(all_sample_pcc).round(4)), 'Cosine Similarity: ', str(np.mean(all_sample_cosine_similarity).round(4)))
    print('By sample: ', 'MSE: ', str(np.mean(all_feature_mse).round(4)), 'MAE: ', str(np.mean(all_feature_mae).round(4)), 'R2: ', str(np.mean(all_feature_r2).round(4)), 'PCC: ', str(np.mean(all_feature_pcc).round(4)), 'Cosine Similarity: ', str(np.mean(all_feature_cosine_similarity).round(4)))
            



