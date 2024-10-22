import os
import random
import pooch
import torch
import esm
import pytest
from esme.esm import ESM2


random.seed(31)
torch.manual_seed(31)

fasta_path = 'tests/data/test.fa'
fai_path = 'tests/data/test.fa.fai'

esm2_8M_url = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt"
esm2_8M_model_path = 'tests/data/esm2_t6_8M_UR50D.pt'
esm2e_8M_path = 'tests/data/8M.safetensors'

if not os.path.exists(esm2_8M_model_path):
    pooch.retrieve(
        url=esm2_8M_url,
        known_hash='md5:8039fc9cee7f71cd2633b13b5a38ff50',
        path=os.path.dirname(esm2_8M_model_path),
        fname=os.path.basename(esm2_8M_model_path)
    )
    pooch.retrieve(
        url=esm2_8M_url.replace(
            '.pt', '-contact-regression.pt').replace('/models/', '/regression/'),
        known_hash='md5:49dffe9c8a53216c7d3948c9fba7dc27',
        path=os.path.dirname(esm2_8M_model_path),
        fname=os.path.basename(esm2_8M_model_path).replace(
            '.pt', '-contact-regression.pt')
    )

device = 0

bz = 2
embed_dim = 320
seq_len = 1250
n_heads = embed_dim // 16

p53_human = 'MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAAPPVAPAPAAPTPAAPAPAPSWPLSSSVPSQKTYQGSYGFRLGFLHSGTAKSVTCTYSPALNKMFCQLAKTCPVQLWVDSTPPPGTRVRAMAIYKQSQHMTEVVRRCPHHERCSDSDGLAPPQHLIRVEGNLRVEYLDDRNTFRHSVVVPYEPPEVGSDCTTIHYNYMCNSSCMGGMNRRPILTIITLEDSSGNLLGRNSFEVRVCACPGRDRRTEEENLRKKGEPHHELPPGSTKRALPNNTSSSPQPKKKPLDGEYFTLQIRGRERFEMFRELNEALELKDAQAGKEPGGSRAHSSHLKSKKGQSTSRHKKLMFKTEGPDSD'
calm1_human = 'MADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGYISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVQMMTAK'


@pytest.fixture
def alphabet():
    return esm.data.Alphabet.from_architecture('ESM-1b')


@pytest.fixture
def token(alphabet):
    tokens = list()

    for _ in range(bz - 1):
        slen = random.randint(50, seq_len)
        tokens.append([0] + [
            alphabet.get_idx(i)
            for i in random.choices(alphabet.standard_toks, k=slen)
        ] + [2] + [alphabet.padding_idx] * (seq_len - slen))

    tokens.append([0] + [
        alphabet.get_idx(i)
        for i in random.choices(alphabet.standard_toks, k=seq_len)
    ] + [2])

    return torch.tensor(tokens, device=device)


@pytest.fixture
def token_p53(alphabet):
    return torch.tensor([[0] + [
        alphabet.get_idx(i)
        for i in p53_human
    ] + [2]], device=device)


@pytest.fixture
def esm2_model():
    model, alphabet = esm.pretrained.load_model_and_alphabet(
        esm2_8M_model_path)
    return model.to(device)


@pytest.fixture
def flash_esm2(esm2_model):
    model = ESM2(
        num_layers=esm2_model.num_layers,
        embed_dim=esm2_model.embed_dim,
        attention_heads=esm2_model.attention_heads,
    )
    params = esm2_model.state_dict()
    params = {
        k.replace('_proj', ''): v
        for k, v in params.items()
        if not k.startswith('contact_head')
    }
    missing, unexpected = model.load_state_dict(params, strict=False)
    model = model.to(device=device).to(torch.bfloat16)
    return model
