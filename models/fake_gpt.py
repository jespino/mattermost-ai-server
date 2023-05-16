class FakeGPT:
    def query(self, bot_description, text):
        fakeText = "This is a fake reply, this is not a real text model"
        for token in fakeText.split():
            yield token

if __name__ == '__main__':
    gpt = FakeGPT()
    query = input("> ")
    while query != "quit":
        print("bot> " + " ".join(gpt.query("you are a smart assistant", query)))
        query = input("> ")
