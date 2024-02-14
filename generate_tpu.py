from lib.generation import generate
from lib.llama import Llama
from lib.logits_processing import PresencePenaltyProcessor, TopKSampler, TopPSampler, make_logits_processor
from lib.param_utils import load_params
from lib.multihost_utils import shard_model_params
from lib.seeding import BEST_INTEGER

import time

def load_params_from_disk() -> Llama:
    cpu_device = jax.devices('tpu')[1]
    with jax.default_device(cpu_device):
        params = load_params('llama2-7B.pickle')
        params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    params = shard_model_params(params)
    return params

def main():
    top_k = 6
    # top_p = 0.05
    max_len = 256

    params = load_params_from_disk()
    print('Successfully loaded model parameters!')

    key = rand.key(BEST_INTEGER, impl='rbg')
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    logits_processor = make_logits_processor(
        PresencePenaltyProcessor(penalty=0.05),
        TopKSampler(top_k=top_k),
        # TopPSampler(top_p=top_p),
    )

    sentences =['Tell me about gravity',]

    start = time.time()
    print("\n\n******* Starting Inference *******\n\n")
    key, subkey = rand.split(key)
    generated_sentences = generate(sentences, tokenizer, params, logits_processor, max_len=max_len, key=subkey)
    for sentence in generated_sentences:
        print(sentence, end='\n\n')

    print("\n\n******* Inference Completed *******\n\n")
    end = time.time()
    print("Time taken for inference: ", round(end-start,2), " seconds")

if __name__ == '__main__':
    main()
