from esme.fasta import Fasta


def test_Fasta():
    fasta = Fasta('tests/data/test.fa')
    assert len(fasta) == 16

    seq = fasta[0]
    assert seq == 'MAFSAEDVLKEYDRRRRMEALLLSLYYPNDRKLLDYKEWSPPRVQVECPKAPVEWNNPPS' \
        'EKGLIVGHFSGIKYKGEKAQASEVDVNKMCCWVSKFKDAMRRYQGIQTCKIPGKVLSDLD' \
        'AKIKAYNLTVEGVEGFVRYSRVTKQHVAAFLKELRHSKQYENVNLIHYILTDKRVDIQHL' \
        'EKDLVKDFKALVESAHRMRQGHMINVKYILYQLLKKHGHGPDGPDILTVKTGSKGVLYDD' \
        'SFRKIYTDLGWKFTPL'

    seq = fasta['Q6GZW5']
    assert seq == 'MKMDTDCRHWIVLASVPVLTVLAFKGEGALALAGLLVMAAVAMYRDRTEKKYSAARAPSP' \
        'IAGHKTAYVTDPSAFAAGTVPVYPAPSNMGSDRFEGWVGGVLTGVGSSHLDHRKFAERQL' \
        'VDRREKMVGYGWTKSFF'


def test_Fasta_len():
    fasta = Fasta('tests/data/test.fa')
    assert len(fasta) == 16
