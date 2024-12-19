from huggingface_hub import HfApi


api = HfApi()

model_filenames = {
    '1b': 'esm1b',
    '1v_1': 'esm1v',
    '1v_1': 'esm1v_1',
    '1v_2': 'esm1v_2',
    '1v_3': 'esm1v_3',
    '1v_4': 'esm1v_4',
    '1v_5': 'esm1v_5',
    '650M': 'esm2',
    '8M': 'esm2_8m',
    '35M': 'esm2_35m',
    '150M': 'esm2_150m',
    '650M': 'esm2_650m',
    '3B': 'esm2_3b',
    '15B': 'esm2_15b',
    'c300m': 'esmc_300m',
    'c600m': 'esmc_600m'
}

if 'ens' in set(snakemake.wildcards.keys()):
    model = f'1v_{snakemake.wildcards["ens"]}'
else:
    model = snakemake.wildcards['model']

api.upload_file(
    path_or_fileobj=snakemake.input['model'],
    path_in_repo=f'{model_filenames[model]}.safetensors',
    repo_id='mhcelik/esm-efficient'
)
