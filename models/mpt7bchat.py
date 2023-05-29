import gpt4all
import sys

class Mpt7bChat:
    def __init__(self):
        self.model = gpt4all.GPT4All("ggml-mpt-7b-chat")

    def query(self, messages):
        result = self.model.chat_completion(messages, verbose=True, streaming=True)
        choice = result["choices"][0]
        choice["delta"] = {"content": choice["message"]["content"]}
        yield result

if __name__ == '__main__':
    gpt = Mpt7bChat()
    query = input("> ")
    while query != "quit":
        print("bot> ", end="")
        for token in gpt.query([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": query}]):
            print(token, end="")
            sys.stdout.flush()
        print("")

        query = input("> ")
