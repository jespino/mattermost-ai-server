import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

class GptNeoxChatBase20B:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
        self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", device_map="auto", load_in_8bit=True)

    def query(self, bot_description, messages):
        original_query = "<human>: {}\n<bot>: ok\n<human>: {}\n<bot>: ".format(bot_description, text)
        for message in messages:
            if message["role"] == "user":
                original_query += "<human>: "+message["content"]
            if message["role"] == "system" or message["role"] == "assistant":
                original_query += "<bot>: "+message["content"]

        current_query = original_query
        output_str = ""
        new_tokens = ""
        while True:
            if new_tokens != "":
                yield new_tokens
            inputs = self.tokenizer(current_query, return_tensors='pt').to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=5, do_sample=True, temperature=0.8)
            output_str = self.tokenizer.decode(outputs[0])
            new_tokens = output_str[len(current_query):]
            current_query = output_str
            if "<human>:" in current_query[len(original_query):]:
                yield new_tokens[:new_tokens.find("<human>:")]
                break

if __name__ == '__main__':
    gpt = GptNeoxChatBase20B()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for token in gpt.query("You are a helpful assistant.", query):
            print(token, end="")
            sys.stdout.flush()
        print("")

        query = input("> ")
