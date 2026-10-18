
class TextGenerator:
    def __init__(self):
        self.text = ""

    def add_letter(self, letter):
        if letter and len(letter) == 1 and letter.isalpha():
            self.text += letter.upper()
        return self.text

    def add_space(self):
        if self.text and not self.text.endswith(" "):
            self.text += " "
        return self.text

    def backspace(self):
        self.text = self.text[:-1]
        return self.text

    def clear(self):
        self.text = ""
        return self.text

    def get_text(self):
        return self.text
