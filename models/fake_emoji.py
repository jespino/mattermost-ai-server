import random

class FakeEmoji():
    def __init__(self):
        self.emojis=[
            "heart",
            "heart_eyes",
            "rolling_on_the_floor_laughing",
            "two_hearts",
            "fire",
            "blush",
            "sunglasses",
            "sparkles",
            "blue_heart",
            "kissing_heart",
            "camera",
            "us",
            "sunny",
            "purple_heart",
            "wink",
            "100",
            "grin",
            "christmas_tree"
            "camera_with_flash",
            "stuck_out_tongue_winking_eye",
        ]

    def query(self, prompt):
        return random.choice(self.emojis)

if __name__ == '__main__':
    emojiSelector = FakeEmoji()
    print(emojiSelector.query("Looking forward to Christmas"))
