dataset_path=/home/lwk/data/EfficientGS

scenes="360_v2/bicycle 360_v2/bonsai 360_v2/counter 360_v2/flowers 360_v2/garden 360_v2/kitchen 360_v2/room 360_v2/stump 360_v2/treehill bonsai db/playroom db/drjohnson tandt/train tandt/truck"

for scene in $scenes
do
    scene_path=$dataset_path/$scene
    model_path=outputs/$scene
    if [ ! -d "$model_path" ]; then
        python train.py -s $scene_path -m $model_path --no_gui --eval --sh_degree 3 --use_grad_norm --grad_norm_way s1 --densify_grad_threshold 0.0007 --prune_by_max_weight --prune_way p4 --oneupSHdegree_from_iter 15500 --shs_lr_rate 5 --use_sparse_shs --sparse_rate 0.2
        python render.py -s $scene_path -m $model_path --iteration 30000 --skip_train
        python metrics.py -m $model_path
        python test_fps.py -s $scene_path -m $model_path --iteration 30000
    fi
done
