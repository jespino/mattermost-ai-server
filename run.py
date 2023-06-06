''' Mattermost AI Server entry point'''
import base64
import json
import importlib
from io import BytesIO
import toml
from gevent import monkey
from bottle import post, run, request, response
from bottle import GeventServer

monkey.patch_all()

config = toml.load("config.toml")


def __import_model(name):
    components = name.split(".")
    module = importlib.import_module(".".join(components[0:-1]))
    return getattr(module, components[-1])

gpt_class = __import_model(config["gpt"]["model_class"])
image_class = __import_model(config["image"]["model_class"])

image_generator = image_class(*config["image"].get("params", []))
text_generator = gpt_class(*config["gpt"].get("params", []))


@post("/chat/completions")
def chat_completions():
    '''API endpoint for text generation'''
    secret = config.get("secret", "")
    if secret != "" and request.headers["Authorization"] != f"Bearer {secret}":
        response.status = 403
        return "permission denied"

    data = request.json
    messages = data["messages"]
    if len(messages) == 0:
        response.status = 400
        return "invalid request"

    yield "retry: 100\n\n"

    if "stream" in data and data["stream"]:
        response.content_type = "text/event-stream"
        response.cache_control = "no-cache"
        response.connection = "keep-alive"
        response.transfer_encoding = "chunked"
        for result in text_generator.query(messages):
            yield f"data: {json.dumps(result)}\n\n"
        last_message = json.dumps({"model": data["model"], "choices": [{"finish_reason": "stop"}]})
        yield f'data: {last_message}\n\n'
        yield "data: [DONE]\n\n"
    else:
        text = ""
        for result in text_generator.query(messages):
            text += result["choices"][0]["delta"]["content"]

        response_body = {
            "choices": [
                {"message": {"content": text}},
            ]
        }
        return response_body
    return ""


@post("/images/generations")
def generate_image():
    '''API image generation'''
    secret = config.get("secret", "")
    if secret != "" and request.headers["Authorization"] != f"Bearer {secret}":
        response.status = 403
        return "permission denied"

    data = request.json
    prompt = data["prompt"] or ""
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
    response_body = {"data": []}

    for _ in range(data["n"]):
        image = image_generator.query(prompt, width, height)
        membuf = BytesIO()
        image.save(membuf, format="png")
        response_body["data"].append(
            {"b64_json": base64.b64encode(membuf.getvalue()).decode("ascii")}
        )

    return response_body


run(
    host=config.get("host", "localhost"),
    port=config.get("port", 8090),
    server=GeventServer,
)
