'''Fake implementation of text generator'''
import time
import sys


class FakeGPT:
    '''Fake implementation class of a text generator'''
    def query(self, text):
        '''Fake query implementation, always return the same result'''
        for token in text.split():
            time.sleep(0.1)
            yield {
                "model": "ggml-mpt-7b-chat",
                "choices": [
                    {"delta": {"content": token + " "}},
                ],
            }


if __name__ == "__main__":
    gpt = FakeGPT()
    query = input("> ")
    while query != "quit":
        print("bot>", end=" ")
        for word in gpt.query("This is a fake reply, this is not a real text model"):
            print(word, end=" ")
            sys.stdout.flush()
        print("")
        query = input("> ")
