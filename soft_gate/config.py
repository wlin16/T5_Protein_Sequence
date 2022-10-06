import torch
import numpy as np



SEED = 2022
np.random.seed(SEED)
torch.manual_seed(SEED)

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

# gene_constrain = 'ohe_constrain'
# situation = 'gene_balance'
# mode = 'mode1'

data_path_1 = '../data/imbalance_same_seq/For_Embedding_seq/mode1_for_embed.csv'
data_path_2 = '../data/imbalance_same_seq/For_Embedding_seq/mode2_for_embed.csv'
# pcm1_path = '../data/pcm1_for_embed.csv'

# model and tokenizer
# tokenizer = T5Tokenizer.from_pretrained(
#     'Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
# model = T5EncoderModel.from_pretrained(
#     "Rostlab/prot_t5_xl_half_uniref50-enc").to(device)

# data process config
batch_size = 32
label_num = 2
# model_size = 2088

# train_parameters
n_epochs = 20
learning_rate = 1e-3
early_stop = 10


# embedding path

model_1_embed_csv_path = '../data/imbalance_same_seq/Embedding_results_csv/mode1_embeds'
model_2_embed_csv_path = '../data/imbalance_same_seq/Embedding_results_csv/mode2_embeds'
# pcm1_save_path = '../data'

# model path
model_dir = './models'

# predict_result_path
# model_1_result_path = '../data/imbalance_same_seq/ML_predicted_results/model_1_results'
# model_2_result_path = '../data/imbalance_same_seq/ML_predicted_results/model_2_results'

save_path = '../../data'



