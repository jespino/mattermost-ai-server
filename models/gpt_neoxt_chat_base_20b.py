import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

class GptNeoxChatBase20B:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
        self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", device_map="auto", load_in_8bit=True)

    def query(self, bot_description, messages):
        original_query = "<human>: {}\n<bot>: ok\n".format(bot_description)
        for message in messages:
            if message["role"] == "user":
                original_query += "<human>: {}\n".format(message["content"])
            if message["role"] == "system" or message["role"] == "assistant":
                original_query += "<bot>: {}\n".format(message["content"])
        original_query += "<bot>: "

        current_query = original_query
        output_str = ""
        buffer = ""
        while True:
            inputs = self.tokenizer(current_query, return_tensors='pt').to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=5, do_sample=True, temperature=0.8)
            output_str = self.tokenizer.decode(outputs[0])
            buffer += output_str[len(current_query):]
            current_query = output_str

            if "<human>:" in buffer:
                yield buffer[:buffer.find("<human>:")]
                break

            if len(buffer) > 20:
                yield buffer[0:-10]
                buffer = buffer[-10:]


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

