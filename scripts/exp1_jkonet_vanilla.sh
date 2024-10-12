for potential in beale double_exp rotational relu flat friedman moon ishigami three_hump_camel bohachevsky sphere styblinski_tang oakley_ohagan cross_in_tray holder_table
do
    python data_generator.py --potential $potential --n-particles 2000 --test-ratio 0.5 --split-trajectories
    python train.py --solver jkonet-vanilla --dataset potential_$potential\_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True --wandb
done