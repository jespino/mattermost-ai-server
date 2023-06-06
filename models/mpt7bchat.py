import gpt4all
import sys
import queue
from gevent import threading


class Mpt7bChat:
    def __init__(self):
        self.model = gpt4all.GPT4All("ggml-mpt-7b-chat")

    def query(self, messages):
        q = queue.Queue(10)

        def response_callback(self, token_id, response):
            response = {
                "model": "ggml-mpt-7b-chat",
                "choices": [
                    {"delta": {"content": response.decode("utf-8")}},
                ],
            }
            q.put(response)
            return True

        self.model.model.__class__._response_callback = response_callback

        def work():
            self.model.chat_completion(messages, streaming=True)
            q.put("end")

        threading.Thread(target=work, daemon=True).start()

        while True:
            value = q.get()
            if value == "end":
                break
            yield value


if __name__ == "__main__":
    gpt = Mpt7bChat()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for token in gpt.query(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query},
            ]
        ):
            print(token, end="")
            sys.stdout.flush()
        print("")

        query = input("> ")
