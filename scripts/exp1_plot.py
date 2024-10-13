import os
import wandb
import numpy as np
from tqdm import tqdm
from load_from_wandb import parse_name

os.makedirs('out/lightspeed', exist_ok=True)

api = wandb.Api()

entity = 'passionfruit-ai'
project = 'learning-diffusion-at-ligthspeed'
group_name = 'split-trajectories-ratio_0.5'

runs = api.runs(f'{entity}/{project}', filters={"group": group_name})

per_method_data = {}
max_error = -np.inf
for run in tqdm(runs):
    run_details = parse_name(run.name)
    potential = run_details['potential']
    method = run_details['method']

    if method not in per_method_data:
        per_method_data[method] = {}


    try:
        summary = run.summary
        err = float(summary['error_w_one_ahead'])
        err_std = float(summary['error_w_one_ahead_std'])
        if run.state != 'finished':
            err = np.nan
            err_std = np.nan
        per_method_data[method][potential] = {
            'error_avg': err,
            'error_std': err_std,
        }
        max_error = np.nanmax([max_error, err])

        history = run.history(pandas=False)
        time = np.asarray([entry['time'] for entry in history if 'time' in entry])
        time = np.asarray(time[(time != None)])
        
        per_method_data[method][potential]['time_avg'] = np.average(time)
        per_method_data[method][potential]['time_std'] = np.std(time)
        # per_method_data[method][potential]['all_errors_avg'] = np.asarray([entry['error_w_one_ahead'] for entry in history])
    except Exception as e:
        print(f'Error in {run.name}')
        print(e)

methods = list(per_method_data.keys())

# In the order of the paper
potentials = [
    'styblinski_tang',
    'holder_table',
    'cross_in_tray',
    'oakley_ohagan',
    'moon',
    'ishigami',
    'friedman',
    'sphere',
    'bohachevsky',
    'three_hump_camel',
    'beale',
    'double_exp',
    'relu',
    'rotational',
    'flat'
]

# Save normalized errors
with open(f'out/lightspeed/error.csv', 'w') as file:
    file.write(','.join(['exp'] + [method for method in methods]) + '\n')
    for (i, potential) in enumerate(potentials):
        file.write(','.join([str(i+1)] + [
            str(10 if np.isnan(per_method_data[method][potential]['error_avg']) else per_method_data[method][potential]['error_avg']) 
            for method in methods]) + '\n')

# Save times
with open(f'out/lightspeed/time.csv', 'w') as file:
    file.write("method,median,q1,q3,lw,uw\n")
    for method in methods:
        data = np.asarray([per_method_data[method][potential]['time_avg'] for potential in potentials])
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lw = np.min(data[data >= q1 - 1.5 * iqr])
        uw = np.max(data[data <= q3 + 1.5 * iqr])
        file.write(f"{method},{median},{q1},{q3},{lw},{uw}\n")