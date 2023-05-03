from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

class StableDiffusion():
    def __init__(self, device = 'cpu'):
        self.model_id = "stabilityai/stable-diffusion-2"
        self.device = device
        self.scheduler = EulerDiscreteScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        self.model = StableDiffusionPipeline.from_pretrained(self.model_id, scheduler=self.scheduler).to(self.device)

    def query(self, text):
        return self.model(text).images[0]

if __name__ == '__main__':
    stableDiffusion = StableDiffusion()
    image = stableDiffusion.query("an astronaut riding a horse in the moon")
    image.save("output.png")
