import gpt4all
import sys

class Gpt4All:
    def __init__(self):
        self.model = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy")

    def query(self, bot_description, messages):
        query = [{"role": "user", "content": bot_description}]
        query += messages
        result = self.model.chat_completion(query, verbose=True, streaming=True)
        yield result["choices"][0]["message"]["content"].strip()

if __name__ == '__main__':
    gpt = Gpt4All()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for token in gpt.query("You are a helpful assistant.", [{"role": "user", "content": query}]):
            print(token, end="")
            sys.stdout.flush()
        print("")

        query = input("> ")
