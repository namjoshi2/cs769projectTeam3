import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import redis
from omegaconf import OmegaConf
import time
from tqdm import tqdm
import editdistance
import argparse

# Import helper functions from baseline (assuming these files exist in your dir)
from evaluate_model_helpers import *

# ==========================================
# 1. FAST BCI MODEL DEFINITION (Must match training!)
# ==========================================
class FastBCIModel(nn.Module):
    def __init__(self, neural_dim, n_units, n_days, n_classes, n_layers=2, 
                 rnn_dropout=0.2, input_dropout=0.2, patch_size=0, patch_stride=0):
        super().__init__()
        
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers 
        self.n_days = n_days
        
        # Day-Specific Layers
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_layer_activation = nn.Softsign()
        self.input_dropout = nn.Dropout(input_dropout)

        # Conv1D Downsampling
        self.conv1 = nn.Conv1d(neural_dim, n_units, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(n_units)
        self.act = nn.ReLU()
        
        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size = n_units,
            hidden_size = n_units,
            num_layers = n_layers,
            dropout = rnn_dropout if n_layers > 1 else 0,
            batch_first = True,
            bidirectional = True 
        )

        # Output Projection
        self.out = nn.Linear(n_units * 2, n_classes)

    def forward(self, x, day_idx, states=None, return_state=False):
        # Day adaptation
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        
        # Conv1D
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)

        # GRU
        output, hidden_states = self.gru(x)

        # Classification
        logits = self.out(output)
        
        if return_state:
            return logits, hidden_states
        
        return logits

# ==========================================
# MAIN EVALUATION SCRIPT
# ==========================================

# argument parser for command line arguments
parser = argparse.ArgumentParser(description='Evaluate the FastBCI model on the copy task dataset.')
parser.add_argument('--model_path', type=str, default='../data/t15_pretrained_rnn_baseline',
                    help='Path to the pretrained model directory.')
parser.add_argument('--data_dir', type=str, default='../data/hdf5_data_final',
                    help='Path to the dataset directory.')
parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                    help='Evaluation type: "val" or "test".')
parser.add_argument('--csv_path', type=str, default='../data/t15_copyTaskData_description.csv',
                    help='Path to the CSV file with metadata.')
parser.add_argument('--gpu_number', type=int, default=0,
                    help='GPU number to use. Set to -1 to use CPU.')
args = parser.parse_args()

model_path = args.model_path
data_dir = args.data_dir
eval_type = args.eval_type 

b2txt_csv_df = pd.read_csv(args.csv_path)
model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))

# Set Device
gpu_number = args.gpu_number
if torch.cuda.is_available() and gpu_number >= 0:
    device = torch.device(f'cuda:{gpu_number}')
    print(f'Using {device} for model inference.')
else:
    print('Using CPU for model inference.')
    device = torch.device('cpu')

# ==========================================
# INITIALIZE FASTBCI MODEL
# ==========================================
print("Initializing FastBCI Model...")
model = FastBCIModel(
    neural_dim = model_args['model']['n_input_features'],
    n_units = model_args['model']['n_units'], 
    n_days = len(model_args['dataset']['sessions']),
    n_classes = model_args['dataset']['n_classes'],
    rnn_dropout = model_args['model']['rnn_dropout'],
    input_dropout = model_args['model']['input_network']['input_layer_dropout'],
    n_layers = model_args['model']['n_layers'],
    patch_size = model_args['model']['patch_size'],
    patch_stride = model_args['model']['patch_stride'],
)

# Load Weights
checkpoint_path = os.path.join(model_path, 'checkpoint/best_checkpoint')
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, weights_only=False)

# Clean up keys (handle DataParallel prefix)
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("module.", "").replace("_orig_mod.", "")
    new_state_dict[new_key] = value

model.load_state_dict(new_state_dict)  
model.to(device) 
model.eval()

# Load Data
test_data = {}
total_test_trials = 0
for session in model_args['dataset']['sessions']:
    files = [f for f in os.listdir(os.path.join(data_dir, session)) if f.endswith('.hdf5')]
    if f'data_{eval_type}.hdf5' in files:
        eval_file = os.path.join(data_dir, session, f'data_{eval_type}.hdf5')
        data = load_h5py_file(eval_file, b2txt_csv_df)
        test_data[session] = data
        total_test_trials += len(test_data[session]["neural_features"])
        print(f'Loaded {len(test_data[session]["neural_features"])} {eval_type} trials for session {session}.')

print(f'Total number of {eval_type} trials: {total_test_trials}\n')

# Inference Loop
with tqdm(total=total_test_trials, desc='Predicting phoneme sequences', unit='trial') as pbar:
    for session, data in test_data.items():
        data['logits'] = []
        data['pred_seq'] = []
        input_layer = model_args['dataset']['sessions'].index(session)
        
        for trial in range(len(data['neural_features'])):
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)
            
            # FIXED: Use float16 instead of bfloat16 for RTX 2080 Ti support
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.float16)

            # Standard decoding step helper
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            data['logits'].append(logits)
            pbar.update(1)

# Convert logits to phonemes
for session, data in test_data.items():
    data['pred_seq'] = []
    for trial in range(len(data['logits'])):
        logits = data['logits'][trial][0]
        pred_seq = np.argmax(logits, axis=-1)
        pred_seq = [int(p) for p in pred_seq if p != 0]
        # Remove consecutive duplicates (CTC decoding standard)
        pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]
        pred_seq = [LOGIT_TO_PHONEME[p] for p in pred_seq]
        data['pred_seq'].append(pred_seq)

        # Print samples
        block_num = data['block_num'][trial]
        trial_num = data['trial_num'][trial]
        print(f'Session: {session}, Block: {block_num}, Trial: {trial_num}')
        if eval_type == 'val':
            sentence_label = data['sentence_label'][trial]
            true_seq = data['seq_class_ids'][trial][0:data['seq_len'][trial]]
            true_seq = [LOGIT_TO_PHONEME[p] for p in true_seq]
            print(f'Sentence label:      {sentence_label}')
            print(f'True sequence:       {" ".join(true_seq)}')
        print(f'Predicted Sequence:  {" ".join(pred_seq)}')
        print()

# ------------------------------------------------------------------
# LANGUAGE MODEL RESCORING (REDIS)
# ------------------------------------------------------------------
r = redis.Redis(host='localhost', port=6379, db=0)
r.flushall()

remote_lm_input_stream = 'remote_lm_input'
remote_lm_output_partial_stream = 'remote_lm_output_partial'
remote_lm_output_final_stream = 'remote_lm_output_final'

# timestamps for redis
remote_lm_output_partial_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_output_final_lastEntrySeen = get_current_redis_time_ms(r)
remote_lm_done_resetting_lastEntrySeen = get_current_redis_time_ms(r)

lm_results = {
    'session': [], 'block': [], 'trial': [],
    'true_sentence': [], 'pred_sentence': [],
}

print("Running Language Model Rescoring...")
with tqdm(total=total_test_trials, desc='Running remote language model', unit='trial') as pbar:
    for session in test_data.keys():
        for trial in range(len(test_data[session]['logits'])):
            logits = rearrange_speech_logits_pt(test_data[session]['logits'][trial])[0]

            remote_lm_done_resetting_lastEntrySeen = reset_remote_language_model(r, remote_lm_done_resetting_lastEntrySeen)
            
            # Send to LM
            remote_lm_output_partial_lastEntrySeen, decoded = send_logits_to_remote_lm(
                r, remote_lm_input_stream, remote_lm_output_partial_stream,
                remote_lm_output_partial_lastEntrySeen, logits,
            )

            # Finalize LM
            remote_lm_output_final_lastEntrySeen, lm_out = finalize_remote_lm(
                r, remote_lm_output_final_stream, remote_lm_output_final_lastEntrySeen,
            )

            best_candidate = lm_out['candidate_sentences'][0]

            lm_results['session'].append(session)
            lm_results['block'].append(test_data[session]['block_num'][trial])
            lm_results['trial'].append(test_data[session]['trial_num'][trial])
            
            if eval_type == 'val':
                lm_results['true_sentence'].append(test_data[session]['sentence_label'][trial])
            else:
                lm_results['true_sentence'].append(None)
            
            lm_results['pred_sentence'].append(best_candidate)
            pbar.update(1)

# Calculate WER if validation
if eval_type == 'val':
    total_true_length = 0
    total_edit_distance = 0
    lm_results['edit_distance'] = []
    lm_results['num_words'] = []

    for i in range(len(lm_results['pred_sentence'])):
        true_sentence = remove_punctuation(lm_results['true_sentence'][i]).strip()
        pred_sentence = remove_punctuation(lm_results['pred_sentence'][i]).strip()
        ed = editdistance.eval(true_sentence.split(), pred_sentence.split())

        total_true_length += len(true_sentence.split())
        total_edit_distance += ed

        lm_results['edit_distance'].append(ed)
        lm_results['num_words'].append(len(true_sentence.split()))

        print(f'{lm_results["session"][i]} - Block {lm_results["block"][i]}, Trial {lm_results["trial"][i]}')
        print(f'True sentence:       {true_sentence}')
        print(f'Predicted sentence:  {pred_sentence}')
        print(f'WER: {ed} / {100 * len(true_sentence.split())} = {ed / len(true_sentence.split()):.2f}%')
        print()

    print(f'Total true sentence length: {total_true_length}')
    print(f'Total edit distance: {total_edit_distance}')
    print(f'Aggregate Word Error Rate (WER): {100 * total_edit_distance / total_true_length:.2f}%')

# Save Output
output_file = os.path.join(model_path, f'fastbci_{eval_type}_predicted_sentences_{time.strftime("%Y%m%d_%H%M%S")}.csv')
ids = [i for i in range(len(lm_results['pred_sentence']))]
df_out = pd.DataFrame({'id': ids, 'text': lm_results['pred_sentence']})
df_out.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
