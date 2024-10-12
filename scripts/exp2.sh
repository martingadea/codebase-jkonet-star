for potential in beale oakley_ohagan double_exp rotational relu flat friedman moon ishigami three_hump_camel bohachevsky sphere styblinski_tang  cross_in_tray holder_table
do
    for dim in 10 20 30 40 50
    do
        for n_particles in 2000 5000 10000 15000 20000
        do
            python data_generator.py --potential $potential --n-particles $n_particles --test-ratio 0.5 --split-trajectories --dimension $dim
            python train.py --solver jkonet-star-potential --dataset potential_$potential\_internal_none_beta_0.0_interaction_none_dt_0.01_T_5_dim_$dim\_N_$n_particles\_gmm_10_seed_0_split_0.5_split_trajectories_True --wandb
        done
    done
done