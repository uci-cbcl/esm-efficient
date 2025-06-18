import random
from tqdm import tqdm
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchmetrics.text import Perplexity
from esme.fasta import Fasta
from esme.alphabet import Alphabet3
from esme.variant import MaskMarginDataset
from esme import ESM2, ESMC
from esm.models.esmc import ESMC as _ESMC


model_name = snakemake.wildcards['model']
efficient = model_name.endswith('e') or model_name.startswith('1ve')
device = snakemake.params['device']
max_len = None

if efficient:
    batch_size = 16
    from esme import ESM
    model = ESM.from_pretrained(snakemake.input['model'], device=device)
else:
    batch_size = 1
    if snakemake.wildcards['model'].startswith('c'):
        from esm.tokenization import get_esmc_model_tokenizers

        if snakemake.wildcards['model'] == 'c300m':
            model = _ESMC(
                d_model=960, n_heads=15, n_layers=30,
                tokenizer=get_esmc_model_tokenizers()
            ).to(torch.bfloat16).eval()

        elif snakemake.wildcards['model'] == 'c600m':
            model = _ESMC(
                d_model=1152, n_heads=18, n_layers=36,
                tokenizer=get_esmc_model_tokenizers()
            ).to(torch.bfloat16).eval()

        model.load_state_dict(torch.load(snakemake.input['model']))
    else:
        import fair_esm
        model, _ = fair_esm.pretrained.load_model_and_alphabet(
            snakemake.input['model'])

        if snakemake.wildcards['model'] == '15B':
            max_len = 1024
        elif snakemake.wildcards['model'] == '1b':
            max_len = 1024
        elif snakemake.wildcards['model'].startswith('1v'):
            max_len = 1024

    model = model.to(device)

def predict_pseudoperplexity(model, seq: str, batch_size=32,
                             max_len=None, alphabet=Alphabet3):
    '''
    Predict the pseudo-perplexity of sequence.
    '''
    device = next(model.parameters()).device

    if isinstance(seq, str):
        dl = DataLoader(MaskMarginDataset(seq, max_len=max_len, alphabet=alphabet),
                        batch_size=batch_size, shuffle=False)
    elif isinstance(seq, DataLoader):
        dl = seq
    elif isinstance(seq, MaskMarginDataset):
        dl = DataLoader(seq, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError('seq must be str or DataLoader')

    perplexity = Perplexity().to(device)

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl):
            if isinstance(model, ESM2) or isinstance(model, ESMC): 
                probs = model(batch['token'].to(device), pad_output=True)
            elif isinstance(model, _ESMC):
                token = batch['token'].to(device)
                logits = model(token, token != 1).sequence_logits
                probs = torch.softmax(logits, dim=-1)
            else:
                probs = model(batch['token'].to(device))['logits']

            batch_idx = torch.arange(probs.size(0))
            pos = batch['local_pos']
            wts = batch['wt_token'].unsqueeze(dim=0)

            probs = probs[batch_idx, pos, :].unsqueeze(0).cpu()
            perplexity.update(probs, wts)

    return perplexity.compute().item()

model.eval()

fasta = Fasta(snakemake.input['fasta'])
random.seed(43)
proteins = random.sample([
    row['id']
    for row in fasta.fai
    if row['length'] <= 2500
], snakemake.params['num_proteins'])

perplexities = dict()
proteins = ['Q96DT6', 'Q969L4', 'P28562', 'Q6SPF0', 'A0A087WWA1', 'Q9H078', 'Q9Y2V3', 'Q9UJW8', 'Q14162', 'Q8NDX6', 'Q9UM73', 'P21731', 'O14827', 'Q96AW1', 'O60829', 'A6NI03', 'Q68EM7', 'P16885', 'O15520', 'Q9BRK5', 'P20815', 'Q30KQ8', 'Q02548', 'Q9BV81', 'Q5K131', 'P38117', 'Q8NBD8', 'A0A1B0GTR4', 'Q9NUL7', 'Q8NDX5', 'Q9P0L1', 'Q8TC36', 'Q14657', 'P17948', 'Q9UPW0', 'Q9NPG4', 'P14091', 'A4D250', 'Q641Q2', 'P07384', 'P27469', 'O14815', 'Q96KR6', 'Q8NA82', 'Q86WU2', 'Q99741', 'Q86YL5', 'Q9Y6Z7', 'Q5GH77', 'Q9UMQ6', 'P51800', 'Q9BZD2', 'P51957', 'O43920', 'Q9BU02', 'Q8NHW5', 'Q96D05', 'Q8N8Z6', 'Q6UY09', 'Q8N9H8', 'O60238', 'A8MWA4', 'Q8IUC2', 'Q6IE37', 'Q96DB9', 'Q08ER8', 'Q9H2F9', 'Q9UBN4', 'Q9Y5P2', 'Q13394', 'Q9NS26', 'Q8NGD5', 'Q8IWW6', 'Q9UQB9', 'P23246', 'Q9HD43', 'Q8NF99', 'Q5SZD1', 'P47884', 'Q96C86', 'Q96CB5', 'Q8NFI3', 'Q16864', 'P45984', 'Q16630', 'Q6NUQ4', 'Q8N0T1', 'Q9Y3C5', 'Q6UXR6', 'Q5T9A4', 'P43005', 'P02750', 'Q6P387', 'Q9NR90', 'Q9NXF7', 'Q5JQC9', 'P48645', 'Q13617', 'Q15399', 'Q9C0H9', 'P50053', 'Q3KQZ1', 'Q99743', 'P16234', 'O14647', 'P15289', 'Q6UXS0', 'Q96P67', 'A6NGN4', 'Q99819', 'O15195', 'Q9H7J1', 'Q5T7B8', 'Q9H5U6', 'A6NC05', 'Q8N8Z3', 'P16333', 'Q9NRX1', 'O95848', 'O75150', 'A0A2Z4LIS9', 'Q9HD45', 'Q9BXR6', 'Q2WGN9', 'P59826', 'Q9NZI8', 'A0A1B0GVS7', 'P0CJ70', 'P61970', 'Q06547', 'Q92769', 'O15534', 'P37287', 'Q96MM6', 'Q16602', 'Q969S0', 'Q9HBF5', 'P17655', 'Q96JQ2', 'P51693', 'Q8N283', 'Q86W33', 'Q15649', 'Q9Y2Z4', 'Q9BXU1', 'Q13277', 'P24387', 'Q8N398', 'Q5VVY1', 'Q9C0H6', 'Q8IV36', 'Q6QHC5', 'P62273', 'Q9NWT1', 'P61567', 'Q96N16', 'Q9ULW3', 'Q96FN4', 'Q6NVY1', 'Q30KQ9', 'Q96S82', 'Q9NZH4', 'O14775', 'Q15365', 'Q9Y4C8', 'P24311', 'Q8IWK6', 'Q13283', 'Q86U86', 'P78358', 'Q9HD90', 'P34949', 'Q6UWL6', 'Q5TGY1', 'Q8NDZ4', 'Q71RG4', 'Q7L2H7', 'P01308', 'Q6IB77', 'P0DOX5', 'O00154', 'Q9ULV3', 'Q9BYQ2', 'P0DMW5', 'Q9NRM7', 'P51679', 'Q96DU3', 'Q99457', 'Q6IAN0', 'Q8TBZ2', 'Q86YM7', 'P36896', 'Q14451', 'Q9NRF2', 'Q8N3D4', 'P22888', 'O75310', 'P10155', 'P33992', 'Q6PGP7', 'Q5GAN6', 'H3BPM6', 'A0A075B6S0', 'Q9HBX3', 'Q14134', 'Q3KQV3', 'Q8N3Y7', 'Q15382', 'Q86UD7', 'Q9HBM0', 'Q6ZRK6', 'Q16778', 'P21754', 'Q9NRM1', 'Q08AN1', 'P08758', 'Q9BU20', 'Q99653', 'Q6UWB1', 'Q6ZQY3', 'P0CG01', 'Q96EG3', 'Q6P1N9', 'P30988', 'Q92806', 'O00519', 'Q9NRG4', 'O43865', 'Q6MZT1', 'A6NDZ8', 'Q13938', 'O95502', 'Q76EJ3', 'P60153', 'A1A580', 'P09132', 'P04440', 'Q6UVK1', 'Q9H1R3', 'P30953', 'Q8NCF5', 'Q8NGG1', 'A8K5M9', 'Q86UU1', 'Q7L523', 'Q86T75', 'Q13201', 'Q8IYK4', 'P01854', 'Q86UD0', 'Q96NZ1', 'Q53GL7', 'Q96G25', 'A7KAX9', 'P10145', 'A2A288', 'Q9BZ81', 'Q08188', 'Q96P09', 'O00358', 'Q5JXM2', 'Q9GZP9', 'Q14507', 'Q99879', 'Q9NRF9', 'Q9NYL4', 'Q6ZR54', 'A6NJG2', 'Q96KD3', 'Q9BVG8', 'A0A1B0GWG4', 'Q5UE93', 'O75663', 'P06241', 'Q9UNX9', 'Q6UX73', 'Q15545', 'Q8NHB7', 'Q9BTC0', 'Q8N668', 'P52758', 'Q15773', 'O94768', 'Q16558', 'P49761', 'Q8IV45', 'Q8IYX1', 'P14923', 'O95861', 'Q6ZW49', 'P08670', 'P52815', 'Q96QD8', 'Q9NX01', 'Q8NCE0', 'P43489', 'O43174', 'Q4G176', 'O43760', 'Q4ZJI4', 'P11912', 'P51884', 'Q6NT55', 'Q49SQ1', 'Q14690', 'O96017', 'O14656', 'Q93062', 'A4D0T7', 'Q16610', 'Q9BQS8', 'Q8NGS0', 'P22303', 'Q96MT8', 'P05129', 'Q96FI4', 'Q9Y2C3', 'O43916', 'A8MYJ7', 'Q15746', 'Q9Y231', 'Q92527', 'Q6PK18', 'Q13491', 'Q8N7R0', 'Q2PZI1', 'P29350', 'P53794', 'P19022', 'Q86TW2', 'Q5JQF7', 'P04233', 'Q99807', 'Q9BVP2', 'Q6X784', 'Q9NYZ3', 'Q9H0B3', 'Q9NY97', 'Q6P1K2', 'Q9Y6H3', 'P16220', 'Q9UKZ1', 'Q9NYZ2', 'O96018', 'P13501', 'O94864', 'P13985', 'Q96L93', 'Q52LA3', 'Q96PV4', 'Q9H7X3', 'Q9ULW2', 'O94988', 'P08590', 'Q86U17', 'Q15836', 'Q86TP1', 'Q8TCU6', 'P52848', 'Q9BRY0', 'Q13177', 'Q6N043', 'P55259', 'M0QZC1', 'Q9Y4F9', 'Q96KF7', 'Q9UKA4', 'Q96MN5', 'Q69384', 'Q86TS9', 'Q9HBZ2', 'P58304', 'Q96AY3', 'Q9H3Y0', 'P11049', 'Q96EX2', 'Q6ZTQ4', 'P0DJI8', 'P24468', 'Q7Z388', 'Q9UKV5', 'P48668', 'P40763', 'Q8IWB9', 'P08567', 'Q8N6P7', 'Q6PJ21', 'Q6DHV7', 'Q9NTI7', 'P11717', 'Q9BXB4', 'Q5TAT6', 'Q504Y2', 'Q99674', 'Q5H9U9', 'Q9UKF2', 'Q10587', 'Q9UDV6', 'Q9Y4X3', 'O14936', 'O94810', 'O95260', 'Q969G5', 'O00461', 'Q07092', 'Q96ET8', 'Q9UJ04', 'P35244', 'Q8TBG9', 'Q9BQD7', 'A0A2R8YFM6', 'Q8IV63', 'P12882', 'Q9NXK6', 'Q86X40', 'Q9BTD8', 'Q8N6T0', 'Q5XXA6', 'P62891', 'Q86X52', 'Q9Y2Y0', 'P51508', 'P78536', 'Q9HBT6', 'Q16671', 'Q14257', 'A0A0A0MT36', 'Q16773', 'Q9P0T7', 'O43148', 'Q92754', 'Q9NZT2', 'Q9Y3D5', 'Q96FF7', 'Q7L311', 'Q7Z7B8', 'Q9ULN7', 'Q0VG99', 'P01593', 'Q96J87', 'Q8NGJ4', 'Q99933', 'P0DOY5', 'Q9Y6Q6', 'P0C864', 'Q6NUJ5', 'Q7Z6G3', 'Q8NH03', 'Q9NYU2', 'P63220', 'O15165', 'Q0VDD5', 'Q2MJR0', 'Q9H6K4', 'Q15468', 'O60883', 'A8MQB3', 'O75116', 'O75884', 'O15539', 'Q96HG1', 'Q5FWF7', 'S4R3P1', 'O75362', 'Q3LI61', 'Q8N371', 'Q5JUX0', 'Q9Y255', 'Q96IL0', 'Q9UKD2', 'Q969H8', 'Q5SRR4', 'Q8IZJ3', 'Q9Y285', 'B4DZS4', 'Q6P5R6', 'P48995', 'Q9NQ25', 'Q8NC69', 'Q9NWQ8', 'Q9ULR3', 'Q9NY37', 'P0DTE8', 'Q8TD06', 'P12259', 'Q02779', 'Q5JY77', 'Q5JRS4', 'P22413', 'Q3ZCM7', 'O75820', 'Q8WV15', 'Q9H9J2', 'E9PB15', 'Q9BTA9', 'Q07912', 'P06310', 'Q9H910', 'A6NLP5', 'Q9H825', 'Q9UGI0', 'O95573', 'F5GYI3', 'Q9Y2H1', 'Q494R0', 'P0DP25', 'Q99797', 'Q6IQ26', 'P08034', 'Q13761', 'A6NCS4', 'O95259']
for protein_id in proteins:
    seq = fasta[protein_id]

    if len(seq) > 1024:
        continue

    perplexities[protein_id] = predict_pseudoperplexity(
        model, seq, batch_size=batch_size, max_len=max_len)

    pd.Series(perplexities).to_csv(
        snakemake.output['perplexity'], sep='\t', header=False)
# import random
# from tqdm import tqdm
# import time
# import torch
# import pandas as pd
# from esme.fasta import Fasta
# from esme.variant import predict_pseudoperplexity


# model_name = snakemake.wildcards['model']
# efficient = model_name.endswith('e') or model_name.startswith('1ve')
# device = snakemake.params['device']
# max_len = None

# if efficient:
#     batch_size = 16
#     from esme import ESM
#     model = ESM.from_pretrained(snakemake.input['model'], device=device)
# else:
#     batch_size = 1
#     if snakemake.wildcards['model'].startswith('c'):
#         from esm.models.esmc import ESMC
#         from esm.tokenization import get_esmc_model_tokenizers

#         if snakemake.wildcards['model'] == 'c300m':
#             model = ESMC(
#                 d_model=960, n_heads=15, n_layers=30,
#                 tokenizer=get_esmc_model_tokenizers()
#             )
#         elif snakemake.wildcards['model'] == 'c600m':
#             model = ESMC(
#                 d_model=1152, n_heads=18, n_layers=36,
#                 tokenizer=get_esmc_model_tokenizers()
#             ).eval()

#         model.load_state_dict(torch.load(snakemake.input['model']))
#     else:
#         import fair_esm
#         model, _ = fair_esm.pretrained.load_model_and_alphabet(
#             snakemake.input['model'])

#         if snakemake.wildcards['model'] == '15B':
#             max_len = 1024
#         elif snakemake.wildcards['model'] == '1b':
#             max_len = 1024
#         elif snakemake.wildcards['model'].startswith('1v'):
#             max_len = 1024

#     model = model.to(device)

# def predict_pseudoperplexity(model, seq: str, batch_size=32,
#                              max_len=None, alphabet=Alphabet3):
#     '''
#     Predict the pseudo-perplexity of sequence.
#     '''
#     device = next(model.parameters()).device

#     if isinstance(seq, str):
#         dl = DataLoader(MaskMarginDataset(seq, max_len=max_len, alphabet=alphabet),
#                         batch_size=batch_size, shuffle=False)
#     elif isinstance(seq, DataLoader):
#         dl = seq
#     elif isinstance(seq, MaskMarginDataset):
#         dl = DataLoader(seq, batch_size=batch_size, shuffle=False)
#     else:
#         raise ValueError('seq must be str or DataLoader')

#     perplexity = Perplexity().to(device)

#     model.eval()
#     with torch.no_grad():
#         for batch in tqdm(dl):
#             if isinstance(model, ESM2) or isinstance(model, ESMC): 
#                 probs = model(batch['token'].to(device), pad_output=True)
#             elif isinstance(model, _ESMC):
#                 token = batch['token'].to(device)
#                 logits = model(token, token != 1).sequence_logits
#                 probs = torch.softmax(logits, dim=-1)
#             else:
#                 probs = model(batch['token'].to(device))['logits']

#             batch_idx = torch.arange(probs.size(0))
#             pos = batch['local_pos']
#             wts = batch['wt_token'].unsqueeze(dim=0)

#             probs = probs[batch_idx, pos, :].unsqueeze(0).cpu()
#             perplexity.update(probs, wts)

#     return perplexity.compute().item()

# model.eval()

# fasta = Fasta(snakemake.input['fasta'])
# random.seed(43)
# proteins = random.sample([
#     row['id']
#     for row in fasta.fai
#     if row['length'] <= 2500
# ], snakemake.params['num_proteins'])

# perplexities = dict()

# for protein_id in proteins:
#     seq = fasta[protein_id]

#     if len(seq) > 1024:
#         continue

#     perplexities[protein_id] = predict_pseudoperplexity(
#         model, seq, batch_size=batch_size, max_len=max_len)

#     pd.Series(perplexities).to_csv(
#         snakemake.output['perplexity'], sep='\t', header=False)