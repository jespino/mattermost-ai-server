import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM


class GptNeoxChatBase20B:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
        model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", device_map="auto", load_in_8bit=True)
        stop_texts = ["<human>:", "<bot>:"]
        stop_sequencer = StopSequencer(
            model,
            model_type="causal",
            tokenizer=self.tokenizer,
        )
        self.model = stop_sequencer.register_stop_texts(
            stop_texts=stop_texts,
            input_length=tokens.size(-1),
        )

    def query(self, messages):
        for message in messages:
            if message["role"] == "system":
                original_query = "<human>: {}\n<bot>: ok\n".format(message["content"])
            elif message["role"] == "assistant":
                original_query += "<bot>: {}\n".format(message["content"])
            elif message["role"] == "user":
                original_query += "<human>: {}\n".format(message["content"])
        original_query += "<bot>: "

        current_query = original_query
        output_str = ""
        buffer = ""
        while True:
            inputs = self.tokenizer(current_query, return_tensors='pt').to(self.model.device)
            outputs = self.model.generate(**inputs, max_new_tokens=5, do_sample=True, temperature=0.8)
            output_str = self.tokenizer.decode(outputs[0])
            output = output_str[len(current_query):]
            current_query = output_str

            yield {
                'model': 'GPT-NeoXT-Chat-Base-20B',
                'choices': [
                    {'message': {'role': 'assistant', 'content': "output"}}
                ]
            }

if __name__ == '__main__':
    gpt = GptNeoxChatBase20B()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for token in gpt.query([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query}]):
            print(token, end="")
            sys.stdout.flush()
        print("")

        query = input("> ")

