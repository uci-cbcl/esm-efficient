import os
import random
import pooch
import torch
import pytest
import fair_esm
from esm.models.esmc import ESMC as _ESMC
from esme.esm import ESM2, ESMC


random.seed(31)
torch.manual_seed(31)

fasta_path = 'tests/data/test.fa'
fai_path = 'tests/data/test.fa.fai'

esm2_8M_url = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t6_8M_UR50D.pt"
esm2_8M_model_path = 'tests/data/esm2_t6_8M_UR50D.pt'
esm2e_8M_path = 'tests/data/8M.safetensors'

esmc_300M_url = "https://huggingface.co/EvolutionaryScale/esmc-300m-2024-12/resolve/main/data/weights/esmc_300m_2024_12_v0.pth?download=true"
esmc_300M_model_path = 'tests/data/esmc_300m_2024_12_v0.pth'

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

if not os.path.exists(esmc_300M_model_path):
    pooch.retrieve(
        url=esmc_300M_url,
        known_hash='md5:000cdd4cb3b8e3e4391a884161fa7434',
        path=os.path.dirname(esmc_300M_model_path),
        fname=os.path.basename(esmc_300M_model_path)
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
    return fair_esm.data.Alphabet.from_architecture('ESM-1b')


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
    model, alphabet = fair_esm.pretrained.load_model_and_alphabet(
        esm2_8M_model_path)
    return model.to(torch.bfloat16).to(device)


@pytest.fixture
def esmc_model():
    model = _ESMC(d_model=960, n_heads=15, n_layers=30, tokenizer=None)
    state_dict = torch.load(esmc_300M_model_path)
    model.load_state_dict(state_dict)
    return model.to(torch.bfloat16).to(device)


@pytest.fixture
def flash_esm2(esm2_model):
    model = ESM2(
        num_layers=esm2_model.num_layers,
        embed_dim=esm2_model.embed_dim,
        attention_heads=esm2_model.attention_heads,
    )
    state_dict = {
        k: v
        for k, v in esm2_model.state_dict().items()
        if not k.startswith('contact_head')
    }
    state_dict_new = {
        'embed_tokens.weight': state_dict['embed_tokens.weight'],
        'lm_head.layer_norm.bias': state_dict['lm_head.layer_norm.bias'],
        'lm_head.layer_norm.weight': state_dict['lm_head.layer_norm.weight'],
        'emb_layer_norm_after.weight': state_dict['emb_layer_norm_after.weight'],
        'emb_layer_norm_after.bias': state_dict['emb_layer_norm_after.bias'],
        'lm_head.dense.weight': state_dict['lm_head.dense.weight'],
        'lm_head.dense.bias': state_dict['lm_head.dense.bias'],
        'lm_head.final.weight': state_dict['lm_head.weight'],
        'lm_head.final.bias': state_dict['lm_head.bias'],
    }

    for i in range(6):
        state_dict_new[f'layers.{i}.self_attn.norm.weight'] = state_dict[
            f'layers.{i}.self_attn_layer_norm.weight']
        state_dict_new[f'layers.{i}.self_attn.norm.bias'] = state_dict[
            f'layers.{i}.self_attn_layer_norm.bias']

        for j in ['q', 'k', 'v', 'out']:
            state_dict_new[f'layers.{i}.self_attn.{j}.weight'] = state_dict[
                f'layers.{i}.self_attn.{j}_proj.weight']
            state_dict_new[f'layers.{i}.self_attn.{j}.bias'] = state_dict[
                f'layers.{i}.self_attn.{j}_proj.bias']

        state_dict_new[f'layers.{i}.final.0.weight'] = state_dict[
            f'layers.{i}.final_layer_norm.weight']
        state_dict_new[f'layers.{i}.final.0.bias'] = state_dict[
            f'layers.{i}.final_layer_norm.bias']
        state_dict_new[f'layers.{i}.final.1.weight'] = state_dict[
            f'layers.{i}.fc1.weight']
        state_dict_new[f'layers.{i}.final.1.bias'] = state_dict[
            f'layers.{i}.fc1.bias']
        state_dict_new[f'layers.{i}.final.3.weight'] = state_dict[
            f'layers.{i}.fc2.weight']
        state_dict_new[f'layers.{i}.final.3.bias'] = state_dict[
            f'layers.{i}.fc2.bias']

    missing, unexpected = model.load_state_dict(state_dict_new, strict=False)
    return model.to(device=device)


@pytest.fixture
def flash_esmc(esmc_model):
    # from esm.tokenization import get_esmc_model_tokenizers

    model = ESMC(num_layers=30, embed_dim=960, attention_heads=15).to(
        device).to(torch.bfloat16)
    state_dict = {
        k: v
        for k, v in esmc_model.state_dict().items()
        if not k.startswith('contact_head')
    }
    state_dict_new = {
        'embed_tokens.weight': state_dict['embed.weight'],
        'emb_layer_norm_after.weight': state_dict['transformer.norm.weight'],
        'lm_head.dense.weight': state_dict['sequence_head.0.weight'],
        'lm_head.dense.bias': state_dict['sequence_head.0.bias'],
        'lm_head.layer_norm.weight': state_dict['sequence_head.2.weight'],
        'lm_head.layer_norm.bias': state_dict['sequence_head.2.bias'],
        'lm_head.final.weight': state_dict['sequence_head.3.weight'],
        'lm_head.final.bias': state_dict['sequence_head.3.bias'],
    }

    for i in range(30):
        state_dict_new[f'layers.{i}.self_attn.norm.weight'] = state_dict[f'transformer.blocks.{i}.attn.layernorm_qkv.0.weight']
        state_dict_new[f'layers.{i}.self_attn.norm.bias'] = state_dict[f'transformer.blocks.{i}.attn.layernorm_qkv.0.bias']
        q, k, v = state_dict[f'transformer.blocks.{i}.attn.layernorm_qkv.1.weight'].chunk(
            3, dim=0)
        # state_dict_new[f'layers.{i}.self_attn.q.weight'] = qkv[:960, :]
        # state_dict_new[f'layers.{i}.self_attn.k.weight'] = qkv[960:1920, :]
        # state_dict_new[f'layers.{i}.self_attn.v.weight'] = qkv[1920:, :]
        state_dict_new[f'layers.{i}.self_attn.q.weight'] = q
        state_dict_new[f'layers.{i}.self_attn.k.weight'] = k
        state_dict_new[f'layers.{i}.self_attn.v.weight'] = v

        state_dict_new[f'layers.{i}.self_attn.out.weight'] = state_dict[f'transformer.blocks.{i}.attn.out_proj.weight']
        state_dict_new[f'layers.{i}.self_attn.layernorm_q.weight'] = state_dict[f'transformer.blocks.{i}.attn.q_ln.weight']
        state_dict_new[f'layers.{i}.self_attn.layernorm_k.weight'] = state_dict[f'transformer.blocks.{i}.attn.k_ln.weight']
        state_dict_new[f'layers.{i}.final.0.weight'] = state_dict[f'transformer.blocks.{i}.ffn.0.weight']
        state_dict_new[f'layers.{i}.final.0.bias'] = state_dict[f'transformer.blocks.{i}.ffn.0.bias']

        _act, _weights = state_dict[f'transformer.blocks.{i}.ffn.1.weight'].chunk(
            2, dim=0)
        state_dict_new[f'layers.{i}.final.1.activation.weight'] = _act
        state_dict_new[f'layers.{i}.final.1.fc.weight'] = _weights
        # state_dict_new[f'layers.{i}.final.1.activation.weight'] = state_dict[f'transformer.blocks.{i}.ffn.1.weight'][:2560, :]
        # state_dict_new[f'layers.{i}.final.1.fc.weight'] = state_dict[f'transformer.blocks.{i}.ffn.1.weight'][2560:, :]
        state_dict_new[f'layers.{i}.final.2.weight'] = state_dict[f'transformer.blocks.{i}.ffn.3.weight']

    missing, unexpected = model.load_state_dict(state_dict_new, strict=False)

    return model.to(device).to(torch.bfloat16)
