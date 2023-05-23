import json
import importlib
import toml

from gevent import monkey; monkey.patch_all()
from gevent import sleep

from bottle import post, get, route, run, template, request, response
from bottle import GeventServer
from io import BytesIO

config = toml.load("config.toml")

def import_model(name):
    components = name.split(".")
    module = importlib.import_module(".".join(components[0:-1]))
    return  getattr(module, components[-1])

class Models:
    def __init__(self):
        GPTClass = import_model(config["gpt"]["model_class"])
        ImageClass = import_model(config["image"]["model_class"])
        EmojiClass = import_model(config["emoji"]["model_class"])

        self.imageGenerator = ImageClass(*config["image"].get("params", []))
        self.textGenerator = GPTClass(*config["gpt"].get("params", []))
        self.emojiSelector = EmojiClass(*config["emoji"].get("params", []))

models = Models()

@post('/v1/chat/completions')
def chat_completions():
    data = request.json
    messages = data['messages']
    if len(messages) == 0:
        response.status = 400
        return "invalid request"

    yield 'retry: 100\n\n'

    if data['stream']:
        response.content_type  = 'text/event-stream'
        response.cache_control = 'no-cache'
        response.connection = 'keep-alive'
        response.transfer_encoding = 'chunked'
        for result in models.textGenerator.query(messages):
            yield 'data: {}\n\n'.format(json.dumps(result))
        yield 'data: {}\n\n'.format(json.dumps({"model": data["model"], "choices": [{"finish_reason": "stop"}]}))
        yield 'data: [DONE]\n\n'
    else:
        return "".join(models.textGenerator.query(messages))

@post('/generateImage')
def generate_image():
    data = request.json
    prompt = data['prompt'] or ""
    if prompt == "":
        response.status = 400
        return "prompt not found"
    image = models.imageGenerator.query(prompt)
    membuf = BytesIO()
    image.save(membuf, format="png")
    return membuf.getvalue()

@post('/selectEmoji')
def generate_emoji():
    data = request.json
    prompt = data['prompt'] or ""
    if prompt == "":
        response.status = 400
        return "prompt not found"
    emoji = models.emojiSelector.query(prompt)
    return json.dumps({"response": emoji})

run(host=config.get("host", "localhost"), port=config.get("port", 8090), server=GeventServer)
