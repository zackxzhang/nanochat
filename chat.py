import argparse
import json
import torch                                                                # type: ignore
import os
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer  # type: ignore


def prelude(message: str, conversation: list, max_tokens: int):
    for i in range(0, len(conversation), 2):
        prompt = tokenizer.apply_chat_template(
            conversation[i:],
            add_generation_prompt=True,
            tokenize=False,
        )
        prompt = tokenizer(prompt, return_tensors='pt')
        if len(prompt['input_ids']) <= max_tokens:
            return prompt
    raise SyntaxError


def chatbot(message: str, conversation: list, max_tokens: int):
    conversation.append({'role': 'user', 'content': message})
    prompt = prelude(message, conversation, max_tokens).to(device)
    output = model.generate(
        **prompt,
        streamer=streamer,
        max_length=max_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        repetition_penalty=1.2,
    ).squeeze()
    response = output[prompt['input_ids'].size(-1):]
    response = tokenizer.decode(response)
    conversation.append({'role': 'assistant', 'content': response})
    return response


parser = argparse.ArgumentParser(
    description='nanochat := a minimalist chatbot'
)
parser.add_argument(
    '-r', '--rounds',
    action='store', type=int,
    required=False, default=64,
    metavar='R', help='number of rounds'
)
parser.add_argument(
    '-m', '--model',
    action='store', type=str,
    required=False, default='llama',
    metavar='M', help='language model'
)


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

    args = parser.parse_args()
    model_id = args.model
    max_rounds = args.rounds
    max_tokens = 2048

    with open('config.json', 'r') as f:
        config = json.load(f)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"SYS:  USING DEVICE *{device}*")

    tokenizer = AutoTokenizer.from_pretrained(
        config[model_id],
        local_files_only=True,
    )
    # money patch >>>
    _decode = tokenizer.decode
    def decode(*args, **kwargs):
        kwargs.update(skip_special_tokens=True)
        return _decode(*args, **kwargs)
    tokenizer.decode = decode
    # <<< money patch
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    model = AutoModelForCausalLM.from_pretrained(
        config[model_id],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()

    print("\nSYS:  YOU HAVE TWO *MAGIC* COMMANDS"
          "\n*EXIT*  TERMINATE CHATBOT AND EXIT PROGRAM"
          "\n*RESET* CLEAR HISTORY AND START CONVERSATION AFRESH")
    print("\nSYS:  CHATBOT ONLINE ...")
    print("-"*64)

    conversation: list = list()
    for _ in range(max_rounds):
        message = input("\nUSER: ")
        if message.strip(' *').lower() == 'exit':
            break
        elif message.strip(' *').lower() == 'reset':
            conversation = list()
            print("\nSYS:  HISTORY CLEARED. CONVERSATION RESTARTED ...")
            continue
        else:
            print("\nBOT:  ", end="")
            response = chatbot(message, conversation, max_tokens)

    print("\nBOT:  It's been great talking with you. "
          "Now I gotta go. Goodbye.\n")
    print("-"*64)
    print("SYS:  CHATBOT OFFLINE ...\n")
