# CARTS
CUDA_VISIBLE_DEVICESS=0,1,2,3 python evaluate.py \
--data-path ./data/minif2f_lean4_v4.10.0/default/ \
--prover CARTS \
--model_name reprover \
--ckpt_path "kaiyuy/leandojo-lean4-tacgen-byt5-small" \
--reward_model_path yangxw/CARTS_vf \
--split test \
--num-cpus 4 \
--with-gpus \
--num-theorems 244 \
--num-sampled-tactics 64 \
--exp-id "minif2f_CARTS" \
--k 8 \
--timeout 600  \
--iteration 100


## BFS
# CUDA_VISIBLE_DEVICESS=0,1,2,3 python evaluate.py \
# --data-path ./data/minif2f_lean4_v4.10.0/default/ \
# --prover BFS \
# --model_name reprover \
# --ckpt_path "kaiyuy/leandojo-lean4-tacgen-byt5-small" \
# --split test \
# --num-cpus 4 \
# --with-gpus \
# --num-theorems 244 \
# --num-sampled-tactics 64 \
# --exp-id "minif2f_BFS" \
# --timeout 600  \
# --iteration 100