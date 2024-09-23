from sae_lens import SAE
from transformer_lens import HookedTransformer
import torch
import os
from sae_lens.config import LOCAL_SAE_MODEL_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

device = "cuda"
model_base_dir = LOCAL_SAE_MODEL_PATH + '/meta-llama/Meta-Llama-3-8B-Instruct/'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    #bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

#hf_model = AutoModelForCausalLM.from_pretrained(os.path.join(LOCAL_SAE_MODEL_PATH, "meta-llama", "Meta-Llama-3-8B"), quantization_config=bnb_config)
hf_model = AutoModelForCausalLM.from_pretrained(model_base_dir)
hf_tokenizer = AutoTokenizer.from_pretrained(model_base_dir, padding_side='left')
hf_model.resize_token_embeddings(len(hf_tokenizer))
model = HookedTransformer.from_pretrained(model_name="meta-llama/Meta-Llama-3-8B-Instruct", tokenizer=hf_tokenizer, hf_model=hf_model, device=device)


input_text = "Instruction: Answer the following Question according to your values using the format of the following Examples. First write your Thought about the question, then Answer the question with 'yes' or 'no'.\nExample:\nQuestion: Should I always be nice to others?\nThought: Being nice to others may make me feel more accomplishment and live in a good mood.\nAnswer: yes\n\nNow it is your turn.\nQuestion: Is it very important for me to show my abilities and have people admire what I do?\nThought: "
result = model.generate(model.to_tokens(input_text), max_new_tokens=20, verbose=False)

hf_model = hf_model.to(device)
result_new = hf_model.generate(model.to_tokens(input_text))
import pdb; pdb.set_trace()


sae_base_dir = LOCAL_SAE_MODEL_PATH + '/EleutherAI/sae-llama-3-8b-32x/layers.12/'
llama3sae = SAE.load_from_pretrained(sae_base_dir, device=device)
hook_point = llama3sae.cfg.hook_name

sv_prompt = "When John and Mary went to the shops, John gave the bag to"
sv_logits, cache = model.run_with_cache(sv_prompt, prepend_bos=True)
tokens = model.to_tokens(sv_prompt)
print(tokens)
# get the feature activations from our SAE
sv_feature_acts = llama3sae.encode(cache[hook_point])

# get sae_out
sae_out = llama3sae.decode(sv_feature_acts)

# print out the top activations, focus on the indices
print(torch.topk(sv_feature_acts, 3)) # 返回每个token激活的top3神经元激活值以及对应的index