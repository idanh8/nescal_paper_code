fl_primary_spec = {
    'fixed_gamma_search': [2, 3],       # Mukhoti et al.'s own fixed-gamma range
    'adaptive_gamma_search': [2, 3],
}

method_specs = {
    'ls_alone': {
        'secondary_loss': None, 'primary_losses': ['LS'],
        'search_type': 'candidate_set', 'search_space': {'alpha': [0.05, 0.1]},
    },
    'mmce': {
        'secondary_loss': 'mmce', 'primary_losses': ['XE'],
        'search_type': 'candidate_set', 'search_space': {'lambda_reg': [1, 3, 7, 10, 15, 20]},
    },
    'mdca': {
        'secondary_loss': 'mdca', 'primary_losses': ['XE'],
        'search_type': 'candidate_set', 'search_space': {'lambda_reg': [1, 5, 10, 15, 20, 25]},
    },
    's_avuc': {
        'secondary_loss': 's-avuc', 'primary_losses': ['XE'],
        'search_type': 'sequential', 'search_order': ['theta', 'temp', 'lambda_reg'],
        'search_space': {'theta': [0.1, 0.5, 0.9], 'temp': [0.5, 1.0, 2.0], 'lambda_reg': [0.1, 1.0, 3.0]},
        'secondary_param_keys': ['theta', 'temp'],
    },
    'mbls': {
        'secondary_loss': 'mbls', 'primary_losses': ['XE'],
        'search_type': 'fixed_per_dataset',
        'fixed_params_by_dataset': {'cifar10/resnet50': {'lambda_reg': 0.1, 'margin': 6}},
        'fallback_search_space': {'lambda_reg': [0, 0.05, 0.1, 0.2, 0.3], 'margin': [4, 6, 8, 10, 12]},
        'secondary_param_keys': ['margin'],
    },
}
