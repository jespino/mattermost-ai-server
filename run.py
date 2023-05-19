import json
import importlib
import toml

from bottle import post, get, route, run, template, request
from bottle.ext.websocket import GeventWebSocketServer
from bottle.ext.websocket import websocket
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

@post('/botQuery')
def botQuery():
    data = request.json
    bot_description = data['bot_description']
    messages = data['messages']
    if len(messages) == 0:
        response.status = 400
        return "invalid request"
    answer = models.textGenerator.query(bot_description, messages)
    return json.dumps({"response": " ".join(answer)})

@get('/botQueryStream', apply=[websocket])
def botQueryStream(ws):
    data = json.loads(ws.receive())
    bot_description = data['bot_description']
    messages = data['messages']
    if len(messages) == 0:
        response.status = 400
        return "invalid request"
    response = ""
    for word in models.textGenerator.query(bot_description, messages):
        ws.send(word)
    ws.send('')
    ws.close()
    return

@post('/generateImage')
def generateImage():
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
def generateImage():
    data = request.json
    prompt = data['prompt'] or ""
    if prompt == "":
        response.status = 400
        return "prompt not found"
    emoji = models.emojiSelector.query(prompt)
    return json.dumps({"response": emoji})

run(host=config.get("host", "localhost"), port=config.get("port", 8090), server=GeventWebSocketServer)
