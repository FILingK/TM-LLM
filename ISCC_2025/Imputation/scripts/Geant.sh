export CUDA_VISIBLE_DEVICES="0"
export TOKENIZERS_PARALLELISM="false"


seq_len=96
model="TM-LLM"
percent=100
mask_rate="0.5"
train_epochs="10"
itr="3"
llm_model="gpt2"
sample_num=5000 # number of training samples
Lambda=5

python -u /root/miniconda3/bin/python run.py \
    --train_epochs $train_epochs \
    --itr $itr \
    --task_name "imputation" \
    --is_training "1" \
    --root_path "../datasets/net_traffic/GEANT" \
    --data_path "geant.csv" \
    --model_id "geant_maskrate_${mask_rate}_${model}_samplenum${sample_num}_seq_${seq_len}" \
    --llm_model $llm_model \
    --sample_num $sample_num \
    --data "net_traffic_geant" \
    --seq_len $seq_len \
    --batch_size "80" \
    --learning_rate "0.001" \
    --mlp "1" \
    --d_model "768" \
    --n_heads "4" \
    --d_ff "768" \
    --enc_in "96" \
    --dec_in "529" \
    --c_out "529" \
    --freq "h" \
    --Lambda $Lambda \
    --percent $percent \
    --gpt_layer "6" \
    --model $model \
    --patience "5" \
    --mask_rate $mask_rate
