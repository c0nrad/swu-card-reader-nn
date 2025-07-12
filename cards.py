from PIL import Image
import numpy as np
from typing import List, Tuple

class Card:
    def __init__(self, name: str, image_path: str, aspects: List[str]):
        self.name = name
        self.image_path = image_path
        self.aspects = aspects
        self.image_data = load_image(image_path)

    def isHeroism(self) -> bool:
        return 'heroism' in self.aspects
    
    def isVillainy(self) -> bool:
        return 'villainy' in self.aspects

    def isNeutral(self) -> bool:
        return not self.isHeroism() and not self.isVillainy()
    
    def isAgression(self) -> bool:
        return 'aggression' in self.aspects
    
CROP_SIZE = (300, 300)  # Crop to 300x300 pixels

def load_image(image_path: str) -> np.ndarray:
    im = Image.open(image_path) # Can be many different formats.
    im = im.crop((0, 0, *CROP_SIZE)) # Crop the image to 300x300 pixels.
    return np.array(im.convert('RGB')).reshape((CROP_SIZE[0]*CROP_SIZE[1]*3, 1))

def load_cards_from_csv(csv_path: str = './cards/cards.csv', count:int=50) -> List[Card]:
    import csv
    cards = []
    with open(csv_path, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # Skip header row
        for row in csv_reader:
            card_data = dict(zip(header, row))
            if card_data['cost'] == "NULL" or card_data['power'] == "NULL" or card_data['hp'] == "NULL":
                continue
            if card_data['leader unit ability text'] != "NULL":
                continue
            
            image_path = f"cards/{card_data['set']}-{card_data['cardnumber']}.png"
            cards.append(Card(card_data['name'], image_path, card_data['aspects'].lower().split(',')))
            count -= 1
            if count == 0:
                break
    return cards

