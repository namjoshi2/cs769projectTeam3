import os
# FORCE DISABLE P2P and INFINIBAND to fix NCCL Error 2
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import random
import time
import os
import numpy as np
import math
import pathlib
import logging
import sys
import json
import pickle

from dataset import BrainToTextDataset, train_test_split_indicies
from data_augmentations import gauss_smooth

import torchaudio.functional as F # for edit distance
from omegaconf import OmegaConf

torch.set_float32_matmul_precision('high') 
torch.backends.cudnn.deterministic = True 
torch._dynamo.config.cache_size_limit = 64

# ==========================================
# 1. IMPROVED FAST MODEL (Matches Baseline Input Logic)
# ==========================================
class FastBCIModel(nn.Module):
    def __init__(self, neural_dim, n_units, n_days, n_classes, n_layers=2, 
                 rnn_dropout=0.2, input_dropout=0.2, patch_size=0, patch_stride=0):
        """
        FastBCIModel: Replaces the slow purely recurrent baseline with a CRNN.
        1. Day-specific adaptation (copied from baseline for stability).
        2. Conv1D downsampling (2x speedup).
        3. Bi-directional GRU (better context).
        """
        super().__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days
        
        # --- 1. Day-Specific Layers (COPIED FROM BASELINE) ---
        # Keeping this identical ensures we handle the multi-day drift correctly.
        # Identity init is crucial so we don't scramble the signal at epoch 0.
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_activation = nn.Softsign()
        self.input_dropout = nn.Dropout(input_dropout)

        # --- 2. The Speed Layer: Conv1D with Stride 2 ---
        # This halves the sequence length immediately.
        # We assume input is (Batch, Time, Features).
        self.conv1 = nn.Conv1d(neural_dim, n_units, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(n_units)
        self.act = nn.ReLU()

        self.se = SEBlock(n_units)
        
        # --- 3. Recurrent Layers ---
        # Bidirectional GRU provides better context than the baseline's unidirectional one.
        self.gru = nn.GRU(
            input_size = n_units,
            hidden_size = n_units,
            num_layers = n_layers,
            dropout = rnn_dropout if n_layers > 1 else 0,
            batch_first = True,
            bidirectional = True 
        )

        # Initialize Weights (Best practice from baseline)
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # --- 4. Output Projection ---
        # Input is n_units * 2 because of Bidirectional GRU
        self.out = nn.Linear(n_units * 2, n_classes)
        nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x, day_idx, states=None, return_state=False):
        """
        Matches the signature of the baseline GRUDecoder so inference scripts work.
        """
        # --- Step 1: Day-Specific Adaptation ---
        # (Exact logic from baseline to project days to common latent space)
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        x = self.input_dropout(x)

        # --- Step 2: Convolutional Downsampling ---
        # Permute to (Batch, Features, Time) for Conv1D
        x = x.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.se(x)
        
        # Permute back to (Batch, Time, Features) for RNN
        x = x.permute(0, 2, 1)

        # --- Step 3: RNN ---
        # We ignore passed 'states' usually because Conv1D breaks 1:1 state mapping,
        # but for full sequence training, we let GRU init its own zero states.
        output, hidden_states = self.gru(x)

        # --- Step 4: Classification ---
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

# ==========================================
# TRAINER CLASS (Modified for FastBCI)
# ==========================================

class BrainToTextDecoder_Trainer:
    def __init__(self, args):
        '''
        args : dictionary of training arguments
        '''
        # Trainer fields
        self.args = args
        self.logger = None 
        self.device = None
        self.model = None
        self.optimizer = None
        self.learning_rate_scheduler = None
        self.ctc_loss = None 

        self.best_val_PER = torch.inf 
        self.best_val_loss = torch.inf 

        self.train_dataset = None 
        self.val_dataset = None 
        self.train_loader = None 
        self.val_loader = None 

        self.transform_args = self.args['dataset']['data_transforms']

        # Create output/checkpoint directories
        if args['mode'] == 'train':
            os.makedirs(self.args['output_dir'], exist_ok=False)
        if args['save_best_checkpoint'] or args['save_all_val_steps'] or args['save_final_model']: 
            os.makedirs(self.args['checkpoint_dir'], exist_ok=False)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        for handler in self.logger.handlers[:]:  
            self.logger.removeHandler(handler)
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s')        

        if args['mode']=='train':
            fh = logging.FileHandler(str(pathlib.Path(self.args['output_dir'],'training_log')))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        self.logger.addHandler(sh)

        # --- DEVICE CONFIGURATION (FIXED FOR 4 GPUs) ---
        if torch.cuda.is_available():
            # If CUDA_VISIBLE_DEVICES sets only 1 GPU, python sees only 1 GPU.
            # In that case, we MUST use cuda:0, regardless of what args say.
            if torch.cuda.device_count() == 1:
                self.logger.info("Only 1 GPU visible. Forcing device to 'cuda:0'.")
                self.device = torch.device("cuda:0")
            elif torch.cuda.device_count() > 1:
                self.logger.info(f"Multi-GPU detected: {torch.cuda.device_count()} GPUs. Using 'cuda:0' as master.")
                self.device = torch.device("cuda:0")
            else:
                # Fallback (should typically not happen if is_available is True)
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.logger.info(f'Using master device: {self.device}')

        if self.args['seed'] != -1:
            np.random.seed(self.args['seed'])
            random.seed(self.args['seed'])
            torch.manual_seed(self.args['seed'])

        # --- MODEL INITIALIZATION ---
        self.model = FastBCIModel(
            neural_dim = self.args['model']['n_input_features'],
            n_units = self.args['model']['n_units'],
            n_days = len(self.args['dataset']['sessions']),
            n_classes  = self.args['dataset']['n_classes'],
            rnn_dropout = self.args['model']['rnn_dropout'], 
            input_dropout = self.args['model']['input_network']['input_layer_dropout'], 
            n_layers = self.args['model']['n_layers']
        )

        # WRAP MODEL IN DATAPARALLEL
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        # MOVE TO DEVICE AFTER WRAPPING
        self.model.to(self.device)

        self.logger.info(f"Initialized FastBCIModel")
        
        # Determine parameter count (handle DataParallel wrapper)
        if isinstance(self.model, nn.DataParallel):
            total_params = sum(p.numel() for p in self.model.module.parameters())
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            
        self.logger.info(f"Model has {total_params:,} parameters")

        # Create datasets
        train_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_train.hdf5') for s in self.args['dataset']['sessions']]
        val_file_paths = [os.path.join(self.args["dataset"]["dataset_dir"],s,'data_val.hdf5') for s in self.args['dataset']['sessions']]

        train_trials, _ = train_test_split_indicies(train_file_paths, 0, self.args['dataset']['seed'], None)
        _, val_trials = train_test_split_indicies(val_file_paths, 1, self.args['dataset']['seed'], None)

        with open(os.path.join(self.args['output_dir'], 'train_val_trials.json'), 'w') as f: 
            json.dump({'train' : train_trials, 'val': val_trials}, f)

        feature_subset = self.args['dataset'].get('feature_subset', None)

        # --- DATALOADERS (FIXED: num_workers=0 to avoid Shared Memory crash) ---
        self.train_dataset = BrainToTextDataset(
            trial_indicies = train_trials,
            split = 'train',
            days_per_batch = self.args['dataset']['days_per_batch'],
            n_batches = self.args['num_training_batches'],
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset
            )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size = None, 
            shuffle = self.args['dataset']['loader_shuffle'],
            num_workers = 0, # FORCE 0
            pin_memory = True 
        )

        self.val_dataset = BrainToTextDataset(
            trial_indicies = val_trials, 
            split = 'test',
            days_per_batch = None,
            n_batches = None,
            batch_size = self.args['dataset']['batch_size'],
            must_include_days = None,
            random_seed = self.args['dataset']['seed'],
            feature_subset = feature_subset   
            )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size = None, 
            shuffle = False, 
            num_workers = 0, # FORCE 0
            pin_memory = True 
        )

        self.logger.info("Successfully initialized datasets")

        self.optimizer = self.create_optimizer()

        if self.args['lr_scheduler_type'] == 'linear':
            self.learning_rate_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer = self.optimizer,
                start_factor = 1.0,
                end_factor = self.args['lr_min'] / self.args['lr_max'],
                total_iters = self.args['lr_decay_steps'],
            )
        elif self.args['lr_scheduler_type'] == 'cosine':
            self.learning_rate_scheduler = self.create_cosine_lr_scheduler(self.optimizer)
        else:
            raise ValueError(f"Invalid learning rate scheduler type")
        
        self.ctc_loss = torch.nn.CTCLoss(blank = 0, reduction = 'none', zero_infinity = False)

        if self.args['init_from_checkpoint']:
            self.load_model_checkpoint(self.args['init_checkpoint_path'])

        # Handle freezing layers (must access .module if DataParallel)
        model_ref = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        for name, param in model_ref.named_parameters():
            if not self.args['model']['rnn_trainable'] and 'gru' in name:
                param.requires_grad = False
            elif not self.args['model']['input_network']['input_trainable'] and 'day' in name:
                param.requires_grad = False

    def create_optimizer(self):
        """
        Fixed optimizer creation to prevent parameter overlap.
        Priority:
        1. Day Parameters (specific LR)
        2. Biases (no weight decay)
        3. Everything else (standard weight decay)
        """
        if isinstance(self.model, nn.DataParallel):
            model_ref = self.model.module
        else:
            model_ref = self.model

        # 1. Day Params: Anything with 'day_' in the name (includes day_weights and day_biases)
        day_params = [p for name, p in self.model.named_parameters() if 'day_' in name]

        # 2. Bias Params: Anything with 'bias' in name, BUT EXCLUDING 'day_' params
        # (This fixes the crash by ensuring day_biases are not double-counted here)
        bias_params = [p for name, p in self.model.named_parameters()
                      if 'bias' in name and 'day_' not in name]

        # 3. Other Params: Everything else (weights for GRU, Conv1D, Linear, etc.)
        other_params = [p for name, p in self.model.named_parameters()
                       if 'day_' not in name and 'bias' not in name]

        # Debug print to ensure no overlap (Optional, can be removed)
        self.logger.info(f"Optimizer Groups: Day params: {len(day_params)}, Bias params: {len(bias_params)}, Other params: {len(other_params)}")

        if len(day_params) != 0:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : day_params, 'lr' : self.args['lr_max_day'], 'weight_decay' : self.args['weight_decay_day'], 'group_type' : 'day_layer'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]
        else:
            param_groups = [
                    {'params' : bias_params, 'weight_decay' : 0, 'group_type' : 'bias'},
                    {'params' : other_params, 'group_type' : 'other'}
                ]

        optim = torch.optim.AdamW(
            param_groups,
            lr = self.args['lr_max'],
            betas = (self.args['beta0'], self.args['beta1']),
            eps = self.args['epsilon'],
            weight_decay = self.args['weight_decay'],
            fused = True
        )

        return optim

    def create_cosine_lr_scheduler(self, optim):
        lr_max = self.args['lr_max']
        lr_min = self.args['lr_min']
        lr_decay_steps = self.args['lr_decay_steps']

        lr_max_day =  self.args['lr_max_day']
        lr_min_day = self.args['lr_min_day']
        lr_decay_steps_day = self.args['lr_decay_steps_day']

        lr_warmup_steps = self.args['lr_warmup_steps']
        lr_warmup_steps_day = self.args['lr_warmup_steps_day']

        def lr_lambda(current_step, min_lr_ratio, decay_steps, warmup_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if current_step < decay_steps:
                progress = float(current_step - warmup_steps) / float(max(1, decay_steps - warmup_steps))
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                return max(min_lr_ratio, min_lr_ratio + (1 - min_lr_ratio) * cosine_decay)
            return min_lr_ratio

        if len(optim.param_groups) == 3:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps), 
                lambda step: lr_lambda(step, lr_min_day / lr_max_day, lr_decay_steps_day, lr_warmup_steps_day),
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),
            ]
        elif len(optim.param_groups) == 2:
            lr_lambdas = [
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),
                lambda step: lr_lambda(step, lr_min / lr_max, lr_decay_steps, lr_warmup_steps),
            ]
        else:
            raise ValueError(f"Invalid number of param groups")
        
        return LambdaLR(optim, lr_lambdas, -1)
        
    def load_model_checkpoint(self, load_path):
        checkpoint = torch.load(load_path, weights_only = False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_PER = checkpoint['val_PER'] 
        self.best_val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint.keys() else torch.inf

        self.model.to(self.device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)
        self.logger.info("Loaded model from checkpoint: " + load_path)

    def save_model_checkpoint(self, save_path, PER, loss):
        checkpoint = {
            'model_state_dict' : self.model.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
            'scheduler_state_dict' : self.learning_rate_scheduler.state_dict(),
            'val_PER' : PER,
            'val_loss' : loss
        }
        torch.save(checkpoint, save_path)
        self.logger.info("Saved model to checkpoint: " + save_path)
        with open(os.path.join(self.args['checkpoint_dir'], 'args.yaml'), 'w') as f:
            OmegaConf.save(config=self.args, f=f)

    def apply_spec_augment(self, features):
        """
        Apply SpecAugment (Time Masking + Frequency Masking)
        """
        B, T, C = features.shape
        # Frequency Masking (up to 10%)
        f_mask_param = int(C * 0.1) 
        f0 = np.random.randint(0, f_mask_param)
        f_start = np.random.randint(0, C - f0)
        features[:, :, f_start:f_start+f0] = 0.0

        # Time Masking (up to 10%)
        t_mask_param = int(T * 0.1)
        t0 = np.random.randint(0, t_mask_param)
        t_start = np.random.randint(0, T - t0)
        features[:, t_start:t_start+t0, :] = 0.0
        return features

    def transform_data(self, features, n_time_steps, mode = 'train'):
        data_shape = features.shape
        batch_size = data_shape[0]
        channels = data_shape[-1]

        if mode == 'train':
            # SpecAugment for robustness
            features = self.apply_spec_augment(features)

            if self.transform_args['static_gain_std'] > 0:
                warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim = 0), (batch_size, 1, 1))
                warp_mat += torch.randn_like(warp_mat, device=self.device) * self.transform_args['static_gain_std']
                features = torch.matmul(features, warp_mat)

            if self.transform_args['white_noise_std'] > 0:
                features += torch.randn(data_shape, device=self.device) * self.transform_args['white_noise_std']

            if self.transform_args['constant_offset_std'] > 0:
                features += torch.randn((batch_size, 1, channels), device=self.device) * self.transform_args['constant_offset_std']

            if self.transform_args['random_walk_std'] > 0:
                features += torch.cumsum(torch.randn(data_shape, device=self.device) * self.transform_args['random_walk_std'], dim =self.transform_args['random_walk_axis'])

            if self.transform_args['random_cut'] > 0:
                cut = np.random.randint(0, self.transform_args['random_cut'])
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

        if self.transform_args['smooth_data']:
            features = gauss_smooth(
                inputs = features, 
                device = self.device,
                smooth_kernel_std = self.transform_args['smooth_kernel_std'],
                smooth_kernel_size= self.transform_args['smooth_kernel_size'],
                )
        
        return features, n_time_steps

    def train(self):
        self.model.train()
        train_losses = []
        val_losses = []
        val_PERs = []
        val_results = []
        val_steps_since_improvement = 0
        save_best_checkpoint = self.args.get('save_best_checkpoint', True)
        early_stopping = self.args.get('early_stopping', True)
        early_stopping_val_steps = self.args['early_stopping_val_steps']
        train_start_time = time.time()

        for i, batch in enumerate(self.train_loader):
            self.model.train()
            self.optimizer.zero_grad()
            start_time = time.time() 

            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.float16):
                features, n_time_steps = self.transform_data(features, n_time_steps, 'train')

                # == FASTBCI MATH: Adjust sequence length for Conv1D Stride 2 ==
                # Formula: floor((Time + 1) / 2)
                adjusted_lens = torch.div(n_time_steps + 1, 2, rounding_mode='floor').to(torch.int32)

                logits = self.model(features, day_indicies)

                loss = self.ctc_loss(
                    log_probs = torch.permute(logits.log_softmax(2), [1, 0, 2]),
                    targets = labels,
                    input_lengths = adjusted_lens,
                    target_lengths = phone_seq_lens
                    )
                loss = torch.mean(loss)
            
            loss.backward()

            if self.args['grad_norm_clip_value'] > 0: 
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               max_norm = self.args['grad_norm_clip_value'],
                                               error_if_nonfinite = True,
                                               foreach = True)

            self.optimizer.step()
            self.learning_rate_scheduler.step()
            
            train_step_duration = time.time() - start_time
            train_losses.append(loss.detach().item())

            if i % self.args['batches_per_train_log'] == 0:
                self.logger.info(f'Train batch {i}: ' +
                        f'loss: {(loss.detach().item()):.2f} ' +
                        f'grad norm: {grad_norm:.2f} '
                        f'time: {train_step_duration:.3f}')

            if i % self.args['batches_per_val_step'] == 0 or i == ((self.args['num_training_batches'] - 1)):
                self.logger.info(f"Running test after training batch: {i}")
                start_time = time.time()
                val_metrics = self.validation(loader = self.val_loader, return_logits = self.args['save_val_logits'], return_data = self.args['save_val_data'])
                val_step_duration = time.time() - start_time

                self.logger.info(f'Val batch {i}: ' +
                        f'PER (avg): {val_metrics["avg_PER"]:.4f} ' +
                        f'CTC Loss (avg): {val_metrics["avg_loss"]:.4f} ' +
                        f'time: {val_step_duration:.3f}')
                
                if self.args['log_individual_day_val_PER']:
                    for day in val_metrics['day_PERs'].keys():
                        self.logger.info(f"{self.args['dataset']['sessions'][day]} val PER: {val_metrics['day_PERs'][day]['total_edit_distance'] / val_metrics['day_PERs'][day]['total_seq_length']:0.4f}")

                val_PERs.append(val_metrics['avg_PER'])
                val_losses.append(val_metrics['avg_loss'])
                val_results.append(val_metrics)

                new_best = False
                if val_metrics['avg_PER'] < self.best_val_PER:
                    self.logger.info(f"New best test PER {self.best_val_PER:.4f} --> {val_metrics['avg_PER']:.4f}")
                    self.best_val_PER = val_metrics['avg_PER']
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True
                elif val_metrics['avg_PER'] == self.best_val_PER and (val_metrics['avg_loss'] < self.best_val_loss): 
                    self.logger.info(f"New best test loss {self.best_val_loss:.4f} --> {val_metrics['avg_loss']:.4f}")
                    self.best_val_loss = val_metrics['avg_loss']
                    new_best = True

                if new_best:
                    if save_best_checkpoint:
                        self.logger.info(f"Checkpointing model")
                        self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/best_checkpoint', self.best_val_PER, self.best_val_loss)
                    if self.args['save_val_metrics']:
                        with open(f'{self.args["checkpoint_dir"]}/val_metrics.pkl', 'wb') as f:
                            pickle.dump(val_metrics, f) 
                    val_steps_since_improvement = 0
                else:
                    val_steps_since_improvement +=1

                if self.args['save_all_val_steps']:
                    self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/checkpoint_batch_{i}', val_metrics['avg_PER'])

                if early_stopping and (val_steps_since_improvement >= early_stopping_val_steps):
                    self.logger.info(f'Overall validation PER has not improved in {early_stopping_val_steps} validation steps. Stopping training early at batch: {i}')
                    break
                
        training_duration = time.time() - train_start_time
        self.logger.info(f'Best avg val PER achieved: {self.best_val_PER:.5f}')
        self.logger.info(f'Total training time: {(training_duration / 60):.2f} minutes')

        if self.args['save_final_model']:
            self.save_model_checkpoint(f'{self.args["checkpoint_dir"]}/final_checkpoint_batch_{i}', val_PERs[-1])

        train_stats = {}
        train_stats['train_losses'] = train_losses
        train_stats['val_losses'] = val_losses 
        train_stats['val_PERs'] = val_PERs
        train_stats['val_metrics'] = val_results
        return train_stats

    def validation(self, loader, return_logits = False, return_data = False):
        self.model.eval()
        metrics = {}
        if return_logits: 
            metrics['logits'] = []
            metrics['n_time_steps'] = []
        if return_data: 
            metrics['input_features'] = []

        metrics['decoded_seqs'] = []
        metrics['true_seq'] = []
        metrics['phone_seq_lens'] = []
        metrics['transcription'] = []
        metrics['losses'] = []
        metrics['block_nums'] = []
        metrics['trial_nums'] = []
        metrics['day_indicies'] = []

        total_edit_distance = 0
        total_seq_length = 0

        day_per = {}
        for d in range(len(self.args['dataset']['sessions'])):
            if self.args['dataset']['dataset_probability_val'][d] == 1: 
                day_per[d] = {'total_edit_distance' : 0, 'total_seq_length' : 0}

        for i, batch in enumerate(loader):        
            features = batch['input_features'].to(self.device)
            labels = batch['seq_class_ids'].to(self.device)
            n_time_steps = batch['n_time_steps'].to(self.device)
            phone_seq_lens = batch['phone_seq_lens'].to(self.device)
            day_indicies = batch['day_indicies'].to(self.device)

            day = day_indicies[0].item()
            if self.args['dataset']['dataset_probability_val'][day] == 0: 
                if self.args['log_val_skip_logs']:
                    self.logger.info(f"Skipping validation on day {day}")
                continue
            
            with torch.no_grad():
                with torch.autocast(device_type = "cuda", enabled = self.args['use_amp'], dtype = torch.float16):
                    features, n_time_steps = self.transform_data(features, n_time_steps, 'val')

                    # == FASTBCI MATH: Adjust sequence length for Conv1D Stride 2 ==
                    adjusted_lens = torch.div(n_time_steps + 1, 2, rounding_mode='floor').to(torch.int32)

                    logits = self.model(features, day_indicies)
    
                    loss = self.ctc_loss(
                        torch.permute(logits.log_softmax(2), [1, 0, 2]),
                        labels,
                        adjusted_lens,
                        phone_seq_lens,
                    )
                    loss = torch.mean(loss)

                metrics['losses'].append(loss.cpu().detach().numpy())

                batch_edit_distance = 0 
                decoded_seqs = []
                for iterIdx in range(logits.shape[0]):
                    decoded_seq = torch.argmax(logits[iterIdx, 0 : adjusted_lens[iterIdx], :].clone().detach(),dim=-1)
                    decoded_seq = torch.unique_consecutive(decoded_seq, dim=-1)
                    decoded_seq = decoded_seq.cpu().detach().numpy()
                    decoded_seq = np.array([i for i in decoded_seq if i != 0])

                    trueSeq = np.array(labels[iterIdx][0 : phone_seq_lens[iterIdx]].cpu().detach())
                    batch_edit_distance += F.edit_distance(decoded_seq, trueSeq)
                    decoded_seqs.append(decoded_seq)

            day = batch['day_indicies'][0].item()
            day_per[day]['total_edit_distance'] += batch_edit_distance
            day_per[day]['total_seq_length'] += torch.sum(phone_seq_lens).item()

            total_edit_distance += batch_edit_distance
            total_seq_length += torch.sum(phone_seq_lens)

            if return_logits: 
                metrics['logits'].append(logits.cpu().float().numpy()) 
                metrics['n_time_steps'].append(adjusted_lens.cpu().numpy())
            if return_data: 
                metrics['input_features'].append(batch['input_features'].cpu().numpy()) 

            metrics['decoded_seqs'].append(decoded_seqs)
            metrics['true_seq'].append(batch['seq_class_ids'].cpu().numpy())
            metrics['phone_seq_lens'].append(batch['phone_seq_lens'].cpu().numpy())
            metrics['transcription'].append(batch['transcriptions'].cpu().numpy())
            metrics['losses'].append(loss.detach().item())
            metrics['block_nums'].append(batch['block_nums'].numpy())
            metrics['trial_nums'].append(batch['trial_nums'].numpy())
            metrics['day_indicies'].append(batch['day_indicies'].cpu().numpy())

        avg_PER = total_edit_distance / total_seq_length
        metrics['day_PERs'] = day_per
        metrics['avg_PER'] = avg_PER.item()
        metrics['avg_loss'] = np.mean(metrics['losses'])

        return metrics
