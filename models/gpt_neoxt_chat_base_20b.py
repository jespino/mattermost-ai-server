import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class GptNeoxChatBase20B:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
        self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", device_map="auto", load_in_8bit=True)
        self.conversation = ""

    def start_conversation(self):
        self.conversation = ""

    def query(self, text):
        if text == "debug":
            print(self.conversation)
            return ""

        original_query = self.conversation+"\n<human>: {}\n<bot>: ".format(text)
        current_query = original_query
        output_str = ""
        while "<human>:" not in current_query[len(original_query):]:
            # TODO: Remove the need of tokenizing/detokenizing here (almost sure is not needed and we can speedup it a bit)
            inputs = self.tokenizer(current_query, return_tensors='pt').to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=True, temperature=0.8)
            output_str = self.tokenizer.decode(outputs[0])
            current_query = output_str
        head = current_query[len(original_query):].split("<human>:")[0]
        self.conversation = original_query+head
        return head

if __name__ == '__main__':
    gpt = GptNeoxChatBase20B()
    query = input("> ")
    while query != "quit":
        print("bot> " + gpt.query(query))
        query = input("> ")
