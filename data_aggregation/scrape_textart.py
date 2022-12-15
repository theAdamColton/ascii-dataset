"""
Scrapes ascii art from https://textart.io/

Categories on this website aren't necessarily mutually exclusive
"""

from bs4 import BeautifulSoup
import requests
import urllib.parse
import os

baseurl = "https://textart.io/"
output_dir = "./data/textart/"

categories = [
    "zubat",
    "lion-cub",
    "meerkat",
    "marshmallow",
    "marine",
    "marching",
    "magic-lamp",
    "lotus",
    "lion",
    "microsoft",
    "linux",
    "light-house",
    "lan",
    "lake",
    "ladder",
    "knight",
    "kneeling",
    "metapod",
    "money",
    "kakuna",
    "nude",
    "pawn",
    "pattern",
    "output",
    "optimus-prime",
    "open-book",
    "oddish",
    "obama",
    "ninetails",
    "native-american",
    "nidorino",
    "nidorina",
    "nidoqueen",
    "nidoking",
    "new-year",
    "network",
    "navy",
    "kangaroo",
    "judy",
    "pidgeot",
    "front-view",
    "gore",
    "google",
    "golbat",
    "gloom",
    "giraffe",
    "gibson",
    "gandalf",
    "friday",
    "hand",
    "france",
    "flying",
    "fly",
    "flemingo",
    "flamingo",
    "fish-tank",
    "fearow",
    "gorilla",
    "happy-holidays",
    "jolly-roger",
    "humming-bird",
    "jigglypuff",
    "jetsons",
    "jet",
    "jar-jar-binks",
    "jane",
    "ivysaur",
    "input",
    "human",
    "head",
    "hug",
    "howl-at-moon",
    "howl",
    "horsemen",
    "herd",
    "helmet",
    "hello-kitty",
    "pentagram",
    "pidgeotto",
    "facebook",
    "temple",
    "turkey",
    "tree-house",
    "tree",
    "timon",
    "tiki",
    "the-wall",
    "thank",
    "symbol",
    "ufo",
    "super-man",
    "sunrise",
    "sun",
    "streching",
    "storage",
    "states",
    "stargate",
    "tweet",
    "venusaur",
    "spearow",
    "wink",
    "yin-yan",
    "x-files",
    "world",
    "woody-woodpecker",
    "woodpecker",
    "wizard",
    "witch",
    "windows",
    "vileplume",
    "wigglytuff",
    "weedle",
    "watch",
    "wartortle",
    "warthog",
    "walk",
    "vulpix",
    "squirtle",
    "sorcerers-stone",
    "pidgey",
    "praying",
    "rattata",
    "raticate",
    "raichu",
    "rafting",
    "raft",
    "r2-d2",
    "pumba",
    "potion",
    "rhinosaurus",
    "poster",
    "police-box",
    "playboy",
    "pink-floyd",
    "pikachu",
    "piegon",
    "pieces",
    "rhino",
    "rocker",
    "software",
    "severed-head",
    "sleeping",
    "slackware",
    "six",
    "simba",
    "signature-block",
    "shop",
    "shark",
    "serpent",
    "ronald-reagen",
    "sea-horse",
    "scorpion",
    "satan",
    "sandslash",
    "sandshrew",
    "sail-boat",
    "rook",
    "faggot",
    "lamp",
    "explosion",
    "blastoise",
    "desert",
    "dobbs",
    "bottle",
    "dollar",
    "bookshelf",
    "book",
    "body-part",
    "bob",
    "donkey",
    "dancing",
    "doors",
    "doric",
    "dragon-fly",
    "bells",
    "beetle",
    "beedrill",
    "druling",
    "beavis",
    "bulbasaur",
    "dance",
    "battle",
    "charmeleon",
    "comic",
    "clock",
    "clefairy",
    "clefable",
    "chinese",
    "corinthian",
    "chick",
    "chess-piece",
    "charmander",
    "daemon",
    "charizard",
    "chamber-of-secrets",
    "cataperie",
    "cpu",
    "camelized",
    "butterfree",
    "butt-head",
    "crossbone",
    "currency-note",
    "bears-standing",
    "bishop",
    "coca-cola",
    "at-at",
    "banner",
    "bank",
    "ekans",
    "att",
    "elroy",
    "at-st",
    "emoticons",
    "barney",
    "arbok",
    "aragorn",
    "apocalypse",
    "aircraft-carrier",
    "africa",
    "evil-queen",
    "3m",
    "egg",
    "email",
    "bashful",
    "earth-mover",
    "baseball",
    "eating",
    "water-spray",
    "dr-seuss",
    "queen",
    "five-continents",
    "cap",
    "crane",
    "crab",
    "statue",
    "apple",
    "stonehendge",
    "person",
    "silhoutte",
    "windmill",
    "spip",
    "one-leg",
    "olympic",
    "story",
    "eve",
    "not-ascii",
    "meditation",
    "nidoran",
    "sun-flower",
    "evolve",
    "minion",
    "one-line-art",
    "eden-garden",
    "kitty",
    "bett-boop",
    "moon",
    "sheep",
    "geant",
    "bike",
    "sms-ascii-art",
    "sex",
    "transformers",
    "eiffel-tower",
    "sea",
    "ionic",
    "snowman",
    "sci-fi",
    "swan",
    "santa",
    "boy",
    "kiss",
    "king",
    "ram",
    "dwarf",
    "bridge",
    "genie",
    "turtle",
    "truck",
    "police",
    "chess-set",
    "buddha",
    "bambi",
    "alligator",
    "usa",
    "tadpole",
    "skyscraper",
    "ark",
    "yoda",
    "musical-instrument",
    "multiple",
    "ansi",
    "chess",
    "pepsi",
    "mask",
    "cross",
    "counting-frame",
    "guitar",
    "howling-at-moon",
    "cows",
    "math",
    "flag",
    "large",
    "pyramid",
    "ring",
    "shuttle",
    "howling",
    "odie",
    "lion-king",
    "casper",
    "monkey",
    "snow-white",
    "jesus",
    "monument",
    "bible",
    "funny",
    "drink",
    "airship",
    "flintstones",
    "tardis",
    "unicode",
    "food",
    "xwing",
    "angel",
    "james-bond",
    "slither",
    "donald-duck",
    "swimming",
    "battle-ship",
    "duckling",
    "starfighter",
    "phone",
    "007",
    "panda",
    "indoor",
    "tiger",
    "shoes",
    "whale",
    "chopper",
    "wolf",
    "game",
    "statue-of-liberty",
    "inside",
    "dr-who",
    "church",
    "fighter-jet",
    "map",
    "walking",
    "icon",
    "snake",
    "asterix-obelix",
    "city",
    "side-view",
    "outdoor",
    "europe",
    "batman",
    "dragon-ball-z",
    "charcter",
    "paris",
    "mickey-mouse",
    "harry-potter",
    "parrot",
    "island",
    "face",
    "big",
    "computer-parts",
    "floppy",
    "mouse",
    "finger",
    "good-detail",
    "pirate",
    "finger-gesture",
    "camel",
    "middle-finger",
    "snail",
    "horse",
    "christianity",
    "explosive",
    "sitting",
    "peanut",
    "pig",
    "car",
    "figlet",
    "elephant",
    "garfield",
    "ship",
    "fighter",
    "animaniacs",
    "101-dalmations",
    "dilbert",
    "military-vehicle",
    "text",
    "tank",
    "spooky",
    "baby",
    "duck",
    "serpentine-dragon",
    "serpentine",
    "man",
    "bee",
    "halloween",
    "house",
    "disney",
    "rocket",
    "penguin",
    "place",
    "basket-ball",
    "vehicle",
    "space",
    "spaceship",
    "spider",
    "rose",
    "male",
    "dolphin",
    "cow",
    "aladdin",
    "fish",
    "clothing",
    "lord-of-the-rings",
    "standing",
    "bear",
    "frames",
    "signature-frames",
    "mail-signature",
    "santa-claus",
    "robot",
    "winnie-the-pooh",
    "wings",
    "winged-dragon",
    "droids",
    "sport",
    "bots",
    "edge",
    "frame",
    "train",
    "naked-women",
    "border",
    "owl",
    "eagle",
    "wearable",
    "birthday",
    "naked",
    "occassion",
    "girl",
    "aliens",
    "small-dragon",
    "figure",
    "shape",
    "brand",
    "geometric",
    "logo",
    "pair",
    "couple",
    "blade",
    "toy",
    "simpson",
    "boat",
    "computer",
    "gun",
    "sword",
    "weapon",
    "kids",
    "charactor",
    "ranma",
    "butterfly",
    "scary",
    "flower",
    "dinosaur",
    "calvin-hobbes",
    "castle",
    "women",
    "female",
    "plane",
    "skeleton",
    "death",
    "skull",
    "faces",
    "machine",
    "single-line-art",
    "chicken",
    "one-line",
    "emoticon",
    "snoopy",
    "smiley",
    "heart",
    "christmas",
    "holiday",
    "rabbit",
    "bunny",
    "starwars",
    "star-wars",
    "dragon",
    "medium",
    "manga",
    "cat",
    "frog",
    "dog",
    "people",
    "building",
    "transport",
    "structure",
    "valentine",
    "love",
    "landscape",
    "critters",
    "pokemon",
    "insect",
    "transportation",
    "bird",
    "movie",
    "animal",
    "small",
    "cartoon",
    "object",
    "hand-drawn",
    "character",
]

discovered_arts = set()

for cat in categories:
    i = 1
    running_total_from_category = 0
    while True:
        url = baseurl + "art/tag/{}/{}".format(cat, i)
        mainpage = requests.get(url)
        soup = BeautifulSoup(mainpage.content, features="html.parser")
        ascii_contents = soup.find_all("div", class_="ascii-wrapper")
        if len(ascii_contents) < 1:
            break
        for ascii_content in ascii_contents:
            art = ascii_content.pre.text
            if not art in discovered_arts:
                running_total_from_category += 1
                discovered_arts.add(art)
                path = os.path.join(output_dir, cat)
                os.makedirs(path, exist_ok=True)
                filename = os.path.join(path, "{}.txt".format(running_total_from_category))
                with open(filename, "w") as f:
                    f.write(art)
                print("Saved ", filename)
            else:
                print("Already discovered!")
        i += 1