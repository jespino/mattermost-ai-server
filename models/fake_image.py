'''Fake image generator implementation'''
import numpy
from PIL import Image, ImageDraw


class FakeImage:
    '''Fake image generator class implementation'''
    def query(self, text, width, height):
        '''fake query implementation for image generation'''
        imarray = numpy.random.rand(width, height, 3) * 255
        img = Image.fromarray(imarray.astype("uint8")).convert("RGBA")
        draw = ImageDraw.Draw(img)
        draw.text((1, 1), text, fill=(0, 0, 0))
        return img


if __name__ == "__main__":
    fakeImage = FakeImage()
    image = fakeImage.query("fake image", 255, 255)
    image.save("output.png")
