import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GptNeoxChatBase20B:
    def __init__(self, device = 'cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
        self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", torch_dtype=torch.bfloat16).to(self.device)

    def query(self, text):
        original_query = "<human>: {}\n<bot>:".format(text)
        current_query = original_query
        while "<human>:" not in current_query[len(original_query):]:
            # TODO: Remove the need of tokenizing/detokenizing here (almost sure is not needed and we can speedup it a bit)
            inputs = tokenizer(current_query, return_tensors='pt').to(self.device)
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
            output_str = tokenizer.decode(outputs[0])
            current_query = output_str
        head, _, _ = current_query.partition("<human>:")
        return head

if __name__ == '__main__':
    gpt = GptNeoxChatBase20B()
    gpt.query("Hello!")
