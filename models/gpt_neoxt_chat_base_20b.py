import torch
import sys
from transformers import StoppingCriteria, StoppingCriteriaList, AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from gevent import threading


class StopStringsCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_string in self.stop_strings:
            input_string = self.tokenizer.decode(input_ids[0])
            if input_string.endswith(stop_string):
                return True
        return False

class GptNeoxChatBase20B:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B")
        self.stop_texts = ["<human>:", "<bot>:"]
        self.stopping_criteria = StopStringsCriteria(self.tokenizer, self.stop_texts)
        self.model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", device_map="auto", load_in_8bit=True)

    def query(self, messages):
        query = ""
        for message in messages:
            if message["role"] == "system":
                query = "**{}**\n".format(message["content"].replace("\n", " "))

        for message in messages:
            if message["role"] == "assistant":
                query += "<bot>: {}\n".format(message["content"])
            elif message["role"] == "user":
                query += "<human>: {}\n".format(message["content"])
        query += "<bot>: "

        streamer = TextIteratorStreamer(self.tokenizer, True)

        inputs = self.tokenizer(query, return_tensors='pt').to(self.model.device)
        generation_kwargs = dict(**inputs, max_new_tokens=1000, do_sample=True, temperature=0.8, stopping_criteria=StoppingCriteriaList([self.stopping_criteria]), streamer=streamer)
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if new_text in self.stop_texts:
                break

            yield {
                'model': 'GPT-NeoXT-Chat-Base-20B',
                'choices': [
                    {'message': {'role': 'assistant', 'content': new_text}, 'delta': {'content': new_text}}
                ]
            }

if __name__ == '__main__':
    gpt = GptNeoxChatBase20B()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for response in gpt.query([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query}]):
            print(response['choices'][0]['delta']['content'], end="")
            sys.stdout.flush()
        print("")

        query = input("> ")
