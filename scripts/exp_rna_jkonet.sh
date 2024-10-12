python preprocess_rna_seq.py --n-components 5
python data_generator.py --load-from-file RNA_PCA_5 --test-ratio 0.4

for seed in 0 1 2 3 4
do
    python train.py --dataset RNA_PCA_5 --solver jkonet --seed $seed --wandb
done