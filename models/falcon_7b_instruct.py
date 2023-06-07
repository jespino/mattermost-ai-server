from gevent import threading
import torch
import sys
from transformers import (
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

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



class Falcon7BInstruct:
    def __init__(self):
        model_name = "tiiuae/falcon-7b-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stop_texts = ["<human>:", "<bot>:"]
        self.stopping_criteria = StopStringsCriteria(self.tokenizer, self.stop_texts)
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    def query(self, messages):
        query = ""
        for message in messages:
            if message["role"] == "system":
                query += "{}\n".format(message["content"].replace("\n", " "))

        for message in messages:
            if message["role"] == "assistant":
                query += "<bot>: {}\n".format(message["content"])
            elif message["role"] == "user":
                query += "<human>: {}\n".format(message["content"])
        query += "<bot>: "

        streamer = TextIteratorStreamer(self.tokenizer, True)
        generation_kwargs = dict(
            max_length=1000,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=0.5,
            stopping_criteria=StoppingCriteriaList([self.stopping_criteria]),
            streamer=streamer
        )
        thread = threading.Thread(target=self.pipeline, args=[query], kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            if new_text in self.stop_texts:
                break

            yield {
                "model": "Falcon7binstruct",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": new_text},
                        "delta": {"content": new_text},
                    }
                ],
            }


if __name__ == "__main__":
    falcon = Falcon7BInstruct()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for response in falcon.query(
            [
                {"role": "system", "content": "<Bot> is a helpful assitant"},
                {"role": "user", "content": query},
            ]
        ):
            print(response["choices"][0]["delta"]["content"], end="")
            sys.stdout.flush()
        print("")

        query = input("> ")
