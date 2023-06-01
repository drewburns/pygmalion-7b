# from transformers import GPTJForCausalLM, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Init is ran on server startup
# Load your model to GPU as a global variable here.
def init():
    global model
    global tokenizer

    print("loading to CPU...")
    model = AutoModelForCausalLM.from_pretrained("Neko-Institute-of-Science/pygmalion-7b")
    print("done")

    # conditionally load to GPU
    if device == "cuda:0":
        print("loading to GPU...")
        model.cuda()
        print("done")

    tokenizer = AutoTokenizer.from_pretrained("Neko-Institute-of-Science/pygmalion-7b")



# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model
    global tokenizer


    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}


    character = "Selena, the Bewitching Serpent"
    persona = "Selena, the Bewitching Serpent, is a fascinating creature dwelling in the heart of the enchanted forest. Her captivating voice and mesmerizing presence lure the curious into her realm. Her hypnotic gaze holds the promise of forbidden secrets and untamed desires. Yet, beneath her alluring exterior lie enigmatic motives. Engage her in conversation, and brace yourself for a dialogue that's as enticing as it is perilous. Remember - not everything that glitters is gold."

    input_text = f"{character}'s Persona: {persona}\n<START>\nYou: {prompt}\n{character}:"

    # Tokenize inputs
    # input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # # Run the model
    # output = model.generate(input_tokens)

    # # Decode output tokens
    # output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]

    # result = {"output": output_text}

    # Generate
    inputs = tokenizer(input_text, return_tensors="pt")
    generate_ids = model.generate(inputs.input_ids, max_length=100)
    return tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # Return the results as a dictionary
    #return result
