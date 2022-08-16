import torch
from transformers import T5Tokenizer, T5EncoderModel
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

SEED = 2022

# ==================== Device ====================================
# GPU = 1
# device = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
elif torch.has_mps:
    torch.cuda.manual_seed(2020)
    device = torch.device('mps')
    print('Device name: MPS')
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# ==================== Dataset and Model ====================================
data_path_1 = './data/mode_1_3w.csv'
data_path_2 = './data/mode_2_3w.csv'

# model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(
    'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
model = T5EncoderModel.from_pretrained(
    "Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# data process config
max_seq_len = 300
label_num = 2
model_size = 1024

# train_parameters
n_epoch = 10
learning_rate = 1e-5
early_stop = 400
batch_size = 32

# embedding path
model_1_embed_path = './Embedding_results/model_1_embeds'
model_2_embed_path = './Embedding_results/model_2_embeds'
model_1_embed_csv_path = './Embedding_results_csv/model_1_embeds'
model_2_embed_csv_path = './Embedding_results_csv/model_2_embeds'

# model path
model_dir = './models'

# predict_result_path
model_1_result_path = './ML_predicted_results/model_1_results'
model_2_result_path = './ML_predicted_results/model_2_results'

# parameter for Boosting Tree models
xgb_space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'n_estimators': 180
    }


