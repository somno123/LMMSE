import torch
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

def seed_everything(seed=2025):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed_everything()

model_id = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def generate_response(input_prompt: str, script: str):
    messages = [
        {
            "role": "system",
            "content": "You are a clinical rater evaluating cat rescue Picture descriptions. Assess linguistic features, coherence, and cognitive indicators in each response."
        },
        {
            "role": "user",
            "content": f"""### Instruction:
            {input_prompt}

            ### Target Script:
            {script}
            """
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
    )

    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

speaker_script = """
# LLM response
"""


user_prompt = """
1. Phrase length: observe the length of paragraph.
2. Grammatical form: observe the variety of grammatical construction.
3. Syntax: observe the sentence structure, clause insertion, word order, syntactic complexity.
4. Word-finding: observe the capacity to evoke needed concept names and informational content in the sentences.
5. Pragmatics: observe the ability to use language effectively in context.

Please evaluate the following dialogue based on the description of the cat rescue illustration, rating each of items 1, 2, 3, 4 and 5 defined above using the following scale: excellent, good, fair, poor, or very poor.
"""

if __name__ == "__main__":
    result = generate_response(user_prompt, speaker_script)
    
    print("-" * 30)
    print(f"LLM response:\n{result}")
    print("-" * 30)
    
