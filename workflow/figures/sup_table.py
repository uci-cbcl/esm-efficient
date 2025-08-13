import pandas as pd


writer = pd.ExcelWriter(snakemake.output['excel'])

dfs = list()

for df in snakemake.input['tables'][:-1]:
    dfs.append(pd.read_csv(df))

df = pd.read_csv(snakemake.input['tables'][-1], header=None)
df.columns = ['go_terms']
dfs.append(df)

df_legend = pd.DataFrame({
    'Table': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S12'],
    'Description': [
        "Memory usage for inference with original and efficient ESM2 implementations.",
        "Inference runtimes of original and efficient ESM2 implementations.",
        "Perplexity of the original and efficient ESM2 model implementations.",
        "Correlation between model predictions and experimental deep mutational scanning data for missense variant effects on the ProteinGYM dataset.",
        "Effect of various optimization strategies on memory usage during model training.",
        "Per-epoch training runtime for original and efficient model implementations.",
        "Melting point prediction performance across different models using head-only fine-tuning.",
        "Performance of fine-tuned models on GB1 fitness landscape prediction using head-only and Head + LoRA fine-tuning.",
        "Performance of fine-tuned models on AAV fitness landscape prediction using head-only and Head + LoRA fine-tuning.",
        "Performance of ESM2 models on transcription factor binding prediction, benchmarked against the DeepTFactor baseline using AUROC.",
        "Performance of ESM2 models on transcription factor binding prediction, benchmarked against the DeepTFactor baseline using AUPRC.",
        "Gene Ontology terms used to define the ground truth for the transcription factor prediction task."
    ]
})
df_legend.to_excel(writer, index=False, sheet_name='Legends')

for i, df in enumerate(dfs):
    df.to_excel(writer, index=False, sheet_name=f'Table S{i + 1}')

writer.close()
