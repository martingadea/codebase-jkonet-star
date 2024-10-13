for potential in beale double_exp rotational relu flat  friedman moon ishigami three_hump_camel bohachevsky sphere styblinski_tang oakley_ohagan cross_in_tray holder_table
do
    for interaction in beale double_exp rotational relu flat  friedman moon ishigami three_hump_camel bohachevsky sphere styblinski_tang oakley_ohagan cross_in_tray holder_table
    do
        for beta in 0.0 0.1 0.2
        do
            python data_generator.py --potential $potential --interaction $interaction --n-particles 2000 --test-ratio 0.5 --internal wiener --beta $beta --split-trajectories
            python train.py --solver jkonet-star --dataset potential_$potential\_internal_wiener_beta_$beta\_interaction_$interaction\_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True --wandb
            python train.py --solver jkonet-star-linear --dataset potential_$potential\_internal_wiener_beta_$beta\_interaction_$interaction\_dt_0.01_T_5_dim_2_N_2000_gmm_10_seed_0_split_0.5_split_trajectories_True --wandb
        done
    done
done