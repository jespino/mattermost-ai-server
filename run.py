import json
import importlib
import toml
import base64

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

        self.imageGenerator = ImageClass(*config["image"].get("params", []))
        self.textGenerator = GPTClass(*config["gpt"].get("params", []))

models = Models()

@post('/chat/completions')
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

@post('/images/generations')
def generate_image():
    data = request.json
    prompt = data['prompt'] or ""
    if prompt == "":
        response.status = 400
        return "prompt not found"
    width = 0
    height = 0
    if data["size"] == "256x256":
        width = 256
        height = 256
    elif data["size"] == "512x512":
        width = 512
        height = 512
    elif data["size"] == "1024x1024":
        width = 1024
        height = 1024
    else:
        response.status = 400
        return "invalid size"
    response = {"data": []}

    for idx in range(data["n"]):
        image = models.imageGenerator.query(prompt, width, height)
        membuf = BytesIO()
        image.save(membuf, format="png")
        response["data"].append({"b64_json": base64.b64encode(membuf.getvalue()).decode('ascii')})

    return response

run(host=config.get("host", "localhost"), port=config.get("port", 8090), server=GeventServer)
