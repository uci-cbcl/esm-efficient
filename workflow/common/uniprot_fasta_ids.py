with open(snakemake.input['fasta'], "r") as f_input:
    with open(snakemake.output['fasta'], 'w') as f_output:
        for line in f_input:
            if line.startswith('>'):
                if snakemake.wildcards['uniprot'] == 'uniprotkb':
                    protein_id = line.split(" ")[0].split("|")[1]
                else:
                    protein_id = line.split(' ')[0].split('_')[1]
                f_output.write(f'>{protein_id}\n')
            else:
                f_output.write(line)
