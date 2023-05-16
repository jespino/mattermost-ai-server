import numpy
from PIL import Image


class FakeImage():
    def query(self, text):
        imarray = numpy.random.rand(512,512,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
        return im


if __name__ == '__main__':
    fakeImage = FakeImage()
    image = fakeImage.query("query")
    image.save("output.png")
