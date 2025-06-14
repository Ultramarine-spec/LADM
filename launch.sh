### 1. Calculate CDS for each 32k sample

# Debug Test
# export CUDA_VISIBLE_DEVICES=7
# python score.py \
#     --chunked_data_path "/public/zhangjiajun/huggingface/datasets/monology/pile-LlamaTokenizerFast-32k-truncated" \
#     --model_path "/public/zhangjiajun/jhchen/storage/LongDepend/outputs/LLM/TinyLlama/TinyLlama_v1.1/pile-LlamaTokenizerFast-32k_seed42/checkpoint-5000" \
#     --model_tag "TinyLlama_v1.1/pile-LlamaTokenizerFast-32k_seed42/checkpoint-5000" \
#     --attn_chunk_size 128 \
#     --d_afs 4 \
#     --d_cds 4 \
#     --num_shards 8 \
#     --ablation 'No' \
#     --part_idx 0 \
#     --test

chunked_data_path="/public/zhangjiajun/huggingface/datasets/monology/pile-LlamaTokenizerFast-32k-truncated"
model_path="/public/zhangjiajun/jhchen/storage/LongDepend/outputs/LLM/TinyLlama/TinyLlama_v1.1/pile-LlamaTokenizerFast-32k_seed42/checkpoint-5000"
model_tag="TinyLlama_v1.1/pile-LlamaTokenizerFast-32k_seed42/checkpoint-5000"
attn_chunk_size=128
d_afs=4
d_cds=4
num_gpus=8
ablation='No'

basename=$(basename "$chunked_data_path")
outputs_dir="./log/${basename}/${model_tag}/chunk${attn_chunk_size}-d_afs${d_afs}-d_cds${d_cds}-${ablation}"
mkdir -p ${outputs_dir}
process_batch() {
    local start_idx=$1
    local end_idx=$2

    for i in $(seq $start_idx $end_idx); do
        export CUDA_VISIBLE_DEVICES=$((i % 8))
        echo ${CUDA_VISIBLE_DEVICES}
        nohup python score.py \
            --chunked_data_path ${chunked_data_path} \
            --model_path ${model_path} \
            --model_tag ${model_tag} \
            --attn_chunk_size ${attn_chunk_size} \
            --d_afs ${d_afs} \
            --d_cds ${d_cds} \
            --num_shards ${num_gpus} \
            --ablation ${ablation} \
            --part_idx ${i} > ${outputs_dir}/log${i}.txt 2>&1 &
    done
    wait
}

process_batch 0 $((num_gpus - 1))


### 2. Filter samples with top CDS
python filter.py \
    --original_data_path "/public/zhangjiajun/huggingface/datasets/monology/pile-LlamaTokenizerFast-32k" \
    --target_path "/public/zhangjiajun/jhchen/huggingface/datasets/monology/pile-LlamaTokenizerFast-32k-truncated/TinyLlama_v1.1/pile-LlamaTokenizerFast-32k_seed42/checkpoint-5000/chunk128-d_afs4-d_cds4"