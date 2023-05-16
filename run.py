import json
import importlib
import toml

# from models.stable_diffusion import StableDiffusion
# from models.gpt_neoxt_chat_base_20b import GptNeoxChatBase20B
# from models.gpt4allj import Gpt4AllJ
# from models.twitter_roberta_base_emoji import TwitterRobertaBaseEmoji
# from models.fake_gpt import FakeGPT

from bottle import post, route, run, template, request
from io import BytesIO

def import_model(name):
    components = name.split(".")
    module = importlib.import_module(".".join(components[0:-1]))
    return  getattr(module, components[-1])

class Models:
    def __init__(self):
        config = toml.load("config.toml")
        GPTClass = import_model(config["gpt"]["model_class"])
        ImageClass = import_model(config["image"]["model_class"])
        EmojiClass = import_model(config["emoji"]["model_class"])

        self.imageGenerator = ImageClass(*config["image"].get("params", []))
        self.textGenerator = GPTClass(*config["gpt"].get("params", []))
        self.emojiSelector = EmojiClass(*config["emoji"].get("params", []))

models = Models()

@post('/botQuery')
def index():
    data = request.json
    bot_description = data['bot_description']
    prompt = data['prompt']
    if prompt == "":
        response.status = 400
        return "prompt not found"
    answer = models.textGenerator.query(bot_description, prompt)
    return json.dumps({"response": " ".join(answer)})

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

run(host='localhost', port=8090)

