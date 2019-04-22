from fontTools.ttLib import TTCollection, TTFont
from fontTools.unicode import Unicode
from itertools import chain
import json

if __name__ == "__main__":
    f = TTFont('D:/work/git/CycleGan-handwriting_generation/data/fonts/simhei.ttf')
    cmap = f.getBestCmap()  # look up the encoding
    list_char_of_font = set()
    for char in sorted(cmap):
        list_char_of_font.add(chr(char))
    # print(len(cmap))
    # print(list_char_of_font)

    list_char_of_hw = set()
    c = 0
    with open('labels.json', encoding='utf-8') as f:
        json_load = json.load(f)
        for line in json_load:
            c += 1
            text = json_load[line]
            for char in list(text):
                if char not in list_char_of_hw:
                    list_char_of_hw.add(char)

    
    
    print('len of font: ', len(list_char_of_font))
    print('len of hw: ', len(list_char_of_hw))
    result = list_char_of_font - list_char_of_hw
    print('len list character miss: ',len(result))