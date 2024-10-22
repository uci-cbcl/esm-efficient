import pandas as pd


writer = pd.ExcelWriter(snakemake.output['excel'])

dfs = list()

for df in snakemake.input['tables'][:-1]:
    dfs.append(pd.read_csv(df))

df = pd.read_csv(snakemake.input['tables'][-1], header=None)
df.columns = ['go_terms']
dfs.append(df)

for i, df in enumerate(dfs):
    df.to_excel(writer, index=False, sheet_name=f'Table S{i + 1}')

writer.close()
