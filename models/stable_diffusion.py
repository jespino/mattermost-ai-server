'''Stable diffusion image generator implementation'''
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler


class StableDiffusion:
    '''Stable diffusion image generator class implementation'''
    def __init__(self, device="cpu"):
        self.model_id = "stabilityai/stable-diffusion-2"
        self.device = device
        self.scheduler = EulerDiscreteScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        self.model = StableDiffusionPipeline.from_pretrained(
            self.model_id, scheduler=self.scheduler
        ).to(self.device)

    def query(self, text, width, height):
        return self.model(text).images[0]


if __name__ == "__main__":
    import threading

    def run(device, output, query):
        '''Run one stable diffusion image generation'''
        stable_diffusion = StableDiffusion(device)
        image = stable_diffusion.query(query, 512, 512)
        image.save(output)

    t1 = threading.Thread(
        target=run,
        args=("cuda:0", "output0.png", "an astronaut riding a horse in the moon"),
    )
    t1.start()
    t2 = threading.Thread(
        target=run,
        args=("cuda:1", "output1.png", "an astronaut riding a horse in the moon"),
    )
    t2.start()
    t3 = threading.Thread(
        target=run,
        args=("cuda:2", "output2.png", "an astronaut riding a horse in the moon"),
    )
    t3.start()
    t4 = threading.Thread(
        target=run,
        args=("cuda:3", "output3.png", "an astronaut riding a horse in the moon"),
    )
    t4.start()

    t1.join()
    t2.join()
    t3.join()
    t4.join()
