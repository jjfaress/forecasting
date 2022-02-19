parameters = [
    {
        'name': 'embedDim',
        'type': 'range',
        'bounds': [8, 128],
        'value_type': 'int'
    },
    {
        'name': 'filters',
        'type': 'range',
        'bounds': [8, 64],
        'value_type': 'int'
    },
    {
        'name': 'kernels',
        'type': 'range',
        'bounds': [8, 64],
        'value_type': 'int'
    },
    {
        'name': 'lstm1',
        'type': 'range',
        'bounds': [8, 128],
        'value_type': 'int'
    },
    {
        'name': 'lstm2',
        'type': 'range',
        'bounds': [8, 128],
        'value_type': 'int'
    },
    {
        'name': 'dRate1',
        'type': 'range',
        'bounds': [0.01, 0.08],
        'log_scale': True
    },
    {
        'name': 'dRate2',
        'type': 'range',
        'bounds': [0.01, 0.08],
        'log_scale': True
    },
    {
        'name': 'maxLen',
        'type': 'range',
        'bounds': [8, 1100],
        'value_type': 'int'
    },
    {
        'name': 'dense',
        'type': 'range',
        'bounds': [6, 32],
        'value_type': 'int'
    },
    {
        'name': 'batch_size',
        'type': 'range',
        'bounds': [16, 128],
        'value_type': 'int'
    },
    {
        'name': 'epochs',
        'type': 'range',
        'bounds': [6, 32],
        'value_type': 'int'
    },

]
