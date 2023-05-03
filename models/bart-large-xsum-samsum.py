from transformers import pipeline

class BartLargeXsumSamsum:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="knkarthick/bart-large-xsum-samsum")

    def query(self, text):
        return self.summarizer(text)

if __name__ == '__main__':
    summarize = BartLargeXsumSamsum()
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        contents.append(line)
    conversation = "\n".join(contents)
    print(summarize.query(conversation)[0]['summary_text'])
