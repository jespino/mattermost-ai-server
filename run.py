from models.stable_diffusion import StableDiffusion
from models.gpt_neoxt_chat_base_20b import GptNeoxChatBase20B
from bottle import post, route, run, template, request
from io import BytesIO

class Models:
    def __init__(self):
        self.imageGenerator = StableDiffusion("cpu")
        # self.textGenerator = GptNeoxChatBase20B()

models = Models()

@post('/botQuery')
def index():
    pprint.pprint(request.json)
    data = request.json
    text = data['prompt']
    if prompt == "":
        response.status = 400
        return "prompt not found"
    self.textGenerator.start_conversation()
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

run(host='localhost', port=8090)

