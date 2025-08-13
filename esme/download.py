import huggingface_hub


model_names = {
    'esm1b',
    'esm1v',
    'esm1v_1',
    'esm1v_2',
    'esm1v_3',
    'esm1v_4',
    'esm1v_5',
    'esm2',
    'esm2_8m',
    'esm2_35m',
    'esm2_150m',
    'esm2_650m',
    'esm2_3b',
    'esm2_15b',
    'esmc',
    'esmc_300m',
    'esmc_600m'
}


def download_model(model_name: str, local_dir: str = None):
    '''
    Download a model from the ESM-1b, ESM-1v, ESM-2, or ESM-C family.   

    Args:
        model_name (str): The model name. Must be one of: 
            'esm1b', 'esm1v', 'esm1v_1', 'esm1v_2', 'esm1v_3', 'esm1v_4', 'esm1v_5',
            'esm2_8m', 'esm2_35m', 'esm2_150m', 'esm2_650m', 'esm2_3b', 'esm2_15b',
            'esmc', 'esmc_300m', 'esmc_600m'
        local_dir (str): The directory to save the model to.
    '''
    if model_name not in model_names:
        raise ValueError(
            f'Invalid model name: {model_name}. Must be one of {model_names}'
        )

    if model_name == 'esm1v':
        model_name = 'esm1v_1'
    elif model_name == 'esm2':
        model_name = 'esm2_650m'
    elif model_name == 'esmc':
        model_name = 'esmc_300m'

    return huggingface_hub.hf_hub_download(
        repo_id='mhcelik/esm-efficient',
        filename=f'{model_name}.safetensors',
        local_dir=local_dir,
    )
