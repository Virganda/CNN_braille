class BrailleTranslator:
    def __init__(self):
        # Standard Grade 1 Braille Mapping
        # Dot positions: 1=top-left, 2=mid-left, 3=bot-left, 4=top-right, 5=mid-right, 6=bot-right
        # Pattern string: '123456' — '1' means dot present, '0' means absent
        self.pattern_map = {
            # Letters
            '100000': 'a', '110000': 'b', '100100': 'c', '100110': 'd',
            '100010': 'e', '110100': 'f', '110110': 'g', '110010': 'h',
            '010100': 'i', '010110': 'j', '101000': 'k', '111000': 'l',
            '101100': 'm', '101110': 'n', '101010': 'o', '111100': 'p',
            '111110': 'q', '111010': 'r', '011100': 's', '011110': 't',
            '101001': 'u', '111001': 'v', '010111': 'w', '101101': 'x',
            '101111': 'y', '101011': 'z',

            # Space
            '000000': ' ',

            # Punctuation & special indicators
            '000110': '.',   # period
            '010000': ',',   # comma
            '010010': '!',   # exclamation
            '001000': '\'',  # apostrophe
            '001010': '?',   # question mark
            '001100': ':',   # colon
            '000010': ';',   # semicolon
            '001001': '-',   # hyphen
            '011010': '"',   # open quote
            '001011': '"',   # close quote
            '001111': '#',   # number indicator
            '000001': '^',   # capital indicator (dots 6 only)
        }

        # Reverse map for debugging
        self.char_map = {v: k for k, v in self.pattern_map.items()}

    def translate(self, pattern):
        """
        Maps a 6-bit string pattern to its character.
        Returns '?' for unknown patterns.
        """
        return self.pattern_map.get(pattern, '?')

    # Keep old name as alias for compatibility
    def pattern_to_char(self, pattern):
        return self.translate(pattern)

    def row_to_sentence(self, char_row):
        return "".join(char_row).strip()

    def join_paragraphs(self, sentences):
        return "\n".join(s for s in sentences if s)


if __name__ == "__main__":
    t = BrailleTranslator()
    print(f"Pattern 100000 is: {t.translate('100000')}")
    print(f"Pattern 000001 is: {t.translate('000001')}")  # capital indicator
