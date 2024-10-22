with open(snakemake.input['fasta'], "r") as f_input:
    with open(snakemake.output['fasta'], 'w') as f_output:
        write_line = False
        for line in f_input:
            if line.startswith('>'):
                if "OX=9606" in line:
                    write_line = True
                    protein_id = line.split(" ")[0].split("|")[1]
                    f_output.write(f'>{protein_id}\n')
                else:
                    write_line = False
            elif write_line:
                f_output.write(line)
