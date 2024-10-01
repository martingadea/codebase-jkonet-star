import re
from typing import Dict

def parse_name(run_name: str) -> Dict[str, str]:
    """
    Parses the run name and extracts model configuration details.

    The run name is expected to follow a specific pattern which includes various configuration parameters.
    This function uses regular expressions to extract these parameters and return them in a dictionary.

    Parameters
    ----------
    run_name : str
        The run name string containing the configuration parameters.

    Returns
    -------
    Dict[str, str]
        A dictionary with the extracted configuration details:
        - 'method': The method used in the run.
        - 'potential': Type of potential used.
        - 'internal': Type of internal process used.
        - 'beta': Beta parameter value.
        - 'interaction': Type of interaction used.
        - 'dt': Time step size.
        - 'T': Total time or steps.
        - 'dim': Dimensionality.
        - 'N': Number of samples.
        - 'gmm': Gaussian Mixture Model parameter.
        - 'seed': Random seed used.
        - 'split': Data split parameter.
    """
    # Regular expression to capture the values
    pattern = re.compile(r'(?P<method_value>.*?).potential_(?P<potential_value>.*?)_internal_(?P<internal_value>.*?)_beta_(?P<beta>[0-9.]+)_interaction_(?P<interaction_value>.*?)_dt_(?P<dt>[0-9.]+)_T_(?P<T>[0-9.]+)_dim_(?P<dim>[0-9.]+)_N_(?P<N>[0-9.]+)_gmm_(?P<gmm>[0-9.]+)_seed_(?P<seed>[0-9.]+)(?:_split_(?P<split>[0-9.]+))?')

    # Search the pattern in the input string
    match = pattern.search(run_name)

    run_details = {}

    # Extract the values
    if match:
        run_details['method'] = match.group('method_value')
        run_details['potential'] = match.group('potential_value')
        run_details['internal'] = match.group('internal_value')
        run_details['beta'] = match.group('beta')
        run_details['interaction'] = match.group('interaction_value')
        run_details['dt'] = match.group('dt')
        run_details['T'] = match.group('T')
        run_details['dim'] = match.group('dim')
        run_details['N'] = match.group('N')
        run_details['gmm'] = match.group('gmm')
        run_details['seed'] = match.group('seed')
        run_details['split'] = match.group('split')

    return run_details
