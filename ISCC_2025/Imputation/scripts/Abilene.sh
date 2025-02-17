# Set environment variables
export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

# Set variables
seq_len=96
model="TM-LLM"
percent=100
mask_rate="0.5"
train_epochs="10"
itr="3"
llm_model="gpt2"
sample_num=1000 # number of training samples
Lambda=2


python -u run.py \
    --train_epochs $train_epochs \
    --itr $itr \
    --task_name "imputation" \
    --is_training "1" \
    --root_path "../datasets/net_traffic/Abilene" \
    --data_path "abilene.csv" \
    --model_id "Abilene_maskrate_${mask_rate}_${model}_samplenum${sample_num}_seq_${seq_len}" \
    --sample_num $sample_num \
    --llm_model $llm_model \
    --data "net_traffic_abilene" \
    --seq_len $seq_len \
    --batch_size "96" \
    --learning_rate "0.001" \
    --mlp "1" \
    --d_model "768" \
    --n_heads "4" \
    --d_ff "768" \
    --enc_in "96" \
    --dec_in "144" \
    --c_out "144" \
    --Lambda $Lambda \
    --freq "h" \
    --percent $percent \
    --gpt_layer "6" \
    --model $model \
    --patience "5" \
    --mask_rate $mask_rate
