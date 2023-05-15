import json

from models.stable_diffusion import StableDiffusion
from models.gpt_neoxt_chat_base_20b import GptNeoxChatBase20B
from models.gpt4allj import Gpt4AllJ
from models.twitter_roberta_base_emoji import TwitterRobertaBaseEmoji
from bottle import post, route, run, template, request
from io import BytesIO

class Models:
    def __init__(self):
        self.imageGenerator = StableDiffusion("cpu")
        self.textGenerator = GptNeoxChatBase20B()
        # self.textGenerator = Gpt4AllJ()
        self.emojiSelector = TwitterRobertaBaseEmoji()

models = Models()

@post('/botQuery')
def index():
    data = request.json
    prompt = data['prompt']
    if prompt == "":
        response.status = 400
        return "prompt not found"
    models.textGenerator.start_conversation()
    answer = models.textGenerator.query(prompt)
    return json.dumps({"response": answer})

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

