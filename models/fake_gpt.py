import time
import sys

class FakeGPT:
    def query(self, text):
        fakeText = "This is a fake reply, this is not a real text model"
        for token in fakeText.split():
            time.sleep(0.1)
            yield {
                'model': "ggml-mpt-7b-chat",
                'choices': [
                    {"delta": {"content": token + " "}},
                ],
            }

if __name__ == '__main__':
    gpt = FakeGPT()
    query = input("> ")
    while query != "quit":
        print("bot>", end=' ')
        for word in gpt.query("you are a smart assistant", query):
            print(word, end=' ')
            sys.stdout.flush()
        print('')
        query = input("> ")
