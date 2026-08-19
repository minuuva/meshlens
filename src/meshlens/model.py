"""Load MeshXL with eager attention so attention weights are actually readable.

The stock MeshXL constructor calls `to_bettertransformer()`, which replaces
self-attention with a fused SDPA kernel. That kernel does not materialize the
softmax matrix, so no attention weights can be read out of it. Every experiment
here depends on those weights, so this loader reproduces MeshXL's construction
exactly except that it requests the eager attention implementation and skips the
BetterTransformer conversion.
"""
import torch, logging
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

N_DISCRETE = 128
BOS, EOS, PAD = 128, 129, 130
VOCAB = N_DISCRETE + 3


def load_meshxl(ckpt_path, llm="mesh-xl/mesh-xl-1.3b"):
    config = AutoConfig.from_pretrained(
        llm, n_positions=8192, max_position_embeddings=8192,
        vocab_size=VOCAB, bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD,
    )
    config.word_embed_proj_dim = config.hidden_size
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        llm, config=config, ignore_mismatched_sizes=True, attn_implementation="eager",
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    # MeshXL wraps the causal LM as `transformer.*` inside its nn.Module
    sd = {k[len("transformer."):]: v for k, v in sd.items() if k.startswith("transformer.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    return model, missing, unexpected
