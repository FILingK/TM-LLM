import os
import subprocess
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

seq_len = 96
model = "TM-LLM"
percent = 100
mask_rate = "0.5"
train_epochs ="10"
itr = "3"
llm_model = "gpt2"
sample_num = 1000 #number of training samples
Lambda = 2

command = [
    "/root/miniconda3/bin/python", "run.py",
    "--train_epochs", train_epochs,
    "--itr", itr,
    "--task_name", "imputation",
    "--is_training", "1",
    "--root_path", r"../datasets/net_traffic/Abilene",
    "--data_path", "abilene.csv",
    "--model_id", f"Abilene_maskrate_{mask_rate}_{model}_samplenum{sample_num}_seq_{seq_len}",
    "--sample_num", str(sample_num),
    "--llm_model", llm_model,
    "--data", "net_traffic_abilene",
    "--seq_len", str(seq_len),
    "--batch_size", "96",
    "--learning_rate", "0.001",
    '--mlp', "1",
    "--d_model", "768",
    "--n_heads", "4",
    "--d_ff", "768",
    "--enc_in", "96",     # input of enc_embedding && output of flatten head
    "--dec_in", "144",     # feature
    "--c_out", "144",       # mlp
    "--Lambda", str(Lambda),
    "--freq", "h",
    "--percent", str(percent),
    "--gpt_layer", "6",
    "--model", model,
    "--patience", "5",
    "--mask_rate", mask_rate
]

subprocess.run(command)
