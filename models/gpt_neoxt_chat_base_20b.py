'''GPT Neoxt base chat 20b model implementation'''
import sys
import torch
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)
from gevent import threading


class StopStringsCriteria(StoppingCriteria):
    '''Stop criteria for end stop tokens'''
    def __init__(self, tokenizer, stop_strings):
        super().__init__()
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_string in self.stop_strings:
            input_string = self.tokenizer.decode(input_ids[0])
            if input_string.endswith(stop_string):
                return True
        return False


class GptNeoxChatBase20B:
    '''GPT Neoxt base chat 20b model class'''
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "togethercomputer/GPT-NeoXT-Chat-Base-20B"
        )
        self.stop_texts = ["<human>:", "<bot>:"]
        self.stopping_criteria = StopStringsCriteria(self.tokenizer, self.stop_texts)
        self.model = AutoModelForCausalLM.from_pretrained(
            "togethercomputer/GPT-NeoXT-Chat-Base-20B",
            device_map="auto",
            load_in_8bit=True,
        )

    def query(self, messages):
        '''Query the model'''
        query_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text = message["content"].replace("\n", " ")
                query_text += f'**{message_text}**\n'

        for message in messages:
            if message["role"] == "assistant":
                query_text += f'<bot>: {message["content"]}\n'
            elif message["role"] == "user":
                query_text += f'<human>: {message["content"]}\n'
        query_text += "<bot>: "

        streamer = TextIteratorStreamer(self.tokenizer, True)

        inputs = self.tokenizer(query_text, return_tensors="pt").to(self.model.device)
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=1000,
            do_sample=True,
            temperature=0.8,
            stopping_criteria=StoppingCriteriaList([self.stopping_criteria]),
            streamer=streamer
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if new_text in self.stop_texts:
                break

            yield {
                "model": "GPT-NeoXT-Chat-Base-20B",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": new_text},
                        "delta": {"content": new_text},
                    }
                ],
            }


if __name__ == "__main__":
    gpt = GptNeoxChatBase20B()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for response in gpt.query(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
        ):
            print(response["choices"][0]["delta"]["content"], end="")
            sys.stdout.flush()
        print("")

        query = input("> ")
