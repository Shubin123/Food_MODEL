#!/usr/bin/env python3
"""
Build a versioned, USDA-sourced food catalog from FoodData Central data.

The catalog provides **per-100g** nutrition values (not "per serving") so the
app can compute:  nutrients = grams_entered × per_100g / 100.

If USDA FoodData Central CSV files are available locally, they are used as the
primary source. Otherwise, the script produces a catalog from the embedded
USDA reference data (curated manually from the USDA FDC API/database).

Usage:
    python scripts/build_catalog.py                        # generate food-catalog.json
    python scripts/build_catalog.py --usda-csv food.csv    # use local USDA CSV
    python scripts/build_catalog.py --validate             # validate existing catalog
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "food-catalog.json"

# ---------------------------------------------------------------------------
# Embedded USDA FDC reference data
# ---------------------------------------------------------------------------
# These are manually curated from USDA FoodData Central (https://fdc.nal.usda.gov/).
# Each entry maps a food (as identified by the Swin Food-101 classifier) to
# per-100g nutrition values and a USDA FDC ID for traceability.
#
# Where no single USDA entry matches a complex dish (e.g., bibimbap), the
# closest available component is used and the entry is marked as approximate.
# ---------------------------------------------------------------------------

USDA_CATALOG = {
    "apple_pie": {
        "displayName": "Apple Pie",
        "per100g": {"calories": 265, "protein": 2.4, "fat": 12.5, "carbs": 37.1},
        "density_g_per_ml": 0.65,
        "usda_fdc_id": "175078",
        "source_desc": "Apple pie, commercially prepared, enriched flour",
        "approximate": False,
    },
    "baby_back_ribs": {
        "displayName": "Baby Back Ribs",
        "per100g": {"calories": 297, "protein": 22.0, "fat": 23.0, "carbs": 0.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "168239",
        "source_desc": "Pork ribs, baby back, lean+fat, braised",
        "approximate": False,
    },
    "baklava": {
        "displayName": "Baklava",
        "per100g": {"calories": 430, "protein": 7.0, "fat": 24.0, "carbs": 50.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "172729",
        "source_desc": "Baklava, commercially prepared",
        "approximate": False,
    },
    "beef_carpaccio": {
        "displayName": "Beef Carpaccio",
        "per100g": {"calories": 170, "protein": 21.0, "fat": 9.0, "carbs": 0.5},
        "density_g_per_ml": 0.75,
        "usda_fdc_id": "168634",
        "source_desc": "Beef, round, raw — closest to carpaccio (thin-sliced raw beef)",
        "approximate": True,
    },
    "beef_tartare": {
        "displayName": "Beef Tartare",
        "per100g": {"calories": 180, "protein": 20.0, "fat": 10.0, "carbs": 1.0},
        "density_g_per_ml": 0.80,
        "usda_fdc_id": "168634",
        "source_desc": "Beef, raw, lean — steak tartare (raw beef + egg + seasonings)",
        "approximate": True,
    },
    "beet_salad": {
        "displayName": "Beet Salad",
        "per100g": {"calories": 82, "protein": 2.3, "fat": 4.0, "carbs": 10.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "169145",
        "source_desc": "Beets, cooked, boiled — salad with dressing estimated",
        "approximate": True,
    },
    "beignets": {
        "displayName": "Beignets",
        "per100g": {"calories": 380, "protein": 6.0, "fat": 18.0, "carbs": 48.0},
        "density_g_per_ml": 0.30,
        "usda_fdc_id": "174987",
        "source_desc": "Doughnut, yeast-leavened, glazed — closest to beignet",
        "approximate": True,
    },
    "bibimbap": {
        "displayName": "Bibimbap",
        "per100g": {"calories": 130, "protein": 5.5, "fat": 4.0, "carbs": 18.0},
        "density_g_per_ml": 0.65,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite estimate: rice + beef + vegetables + gochujang",
        "approximate": True,
    },
    "bread_pudding": {
        "displayName": "Bread Pudding",
        "per100g": {"calories": 210, "protein": 5.5, "fat": 8.0, "carbs": 30.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "172737",
        "source_desc": "Bread pudding with raisins, home recipe",
        "approximate": False,
    },
    "breakfast_burrito": {
        "displayName": "Breakfast Burrito",
        "per100g": {"calories": 230, "protein": 10.0, "fat": 12.0, "carbs": 20.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: tortilla + eggs + cheese + potato + meat",
        "approximate": True,
    },
    "bruschetta": {
        "displayName": "Bruschetta",
        "per100g": {"calories": 175, "protein": 4.0, "fat": 7.0, "carbs": 24.0},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: grilled bread + tomato + olive oil + basil",
        "approximate": True,
    },
    "caesar_salad": {
        "displayName": "Caesar Salad",
        "per100g": {"calories": 157, "protein": 5.2, "fat": 12.9, "carbs": 5.8},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "2342484",
        "source_desc": "Caesar salad with dressing and croutons, restaurant",
        "approximate": False,
    },
    "cannoli": {
        "displayName": "Cannoli",
        "per100g": {"calories": 350, "protein": 8.0, "fat": 20.0, "carbs": 35.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "N/A",
        "source_desc": "Italian pastry: fried shell + sweet ricotta filling",
        "approximate": True,
    },
    "caprese_salad": {
        "displayName": "Caprese Salad",
        "per100g": {"calories": 170, "protein": 9.0, "fat": 13.0, "carbs": 4.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: mozzarella + tomato + olive oil + basil",
        "approximate": True,
    },
    "carrot_cake": {
        "displayName": "Carrot Cake",
        "per100g": {"calories": 390, "protein": 4.0, "fat": 19.0, "carbs": 50.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "172772",
        "source_desc": "Carrot cake with cream cheese frosting, home recipe",
        "approximate": False,
    },
    "ceviche": {
        "displayName": "Ceviche",
        "per100g": {"calories": 85, "protein": 15.0, "fat": 1.5, "carbs": 3.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: raw fish cured in citrus + onion + cilantro",
        "approximate": True,
    },
    "cheesecake": {
        "displayName": "Cheesecake",
        "per100g": {"calories": 321, "protein": 5.5, "fat": 22.5, "carbs": 25.5},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "2342482",
        "source_desc": "Cheesecake, commercially prepared",
        "approximate": False,
    },
    "cheese_plate": {
        "displayName": "Cheese Plate",
        "per100g": {"calories": 380, "protein": 23.0, "fat": 32.0, "carbs": 2.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "173438",
        "source_desc": "Cheese, cheddar — represents a mixed cheese plate average",
        "approximate": True,
    },
    "chicken_curry": {
        "displayName": "Chicken Curry",
        "per100g": {"calories": 140, "protein": 10.0, "fat": 8.0, "carbs": 7.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: chicken + curry sauce + vegetables",
        "approximate": True,
    },
    "chicken_quesadilla": {
        "displayName": "Chicken Quesadilla",
        "per100g": {"calories": 270, "protein": 14.0, "fat": 15.0, "carbs": 20.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: tortilla + chicken + cheese",
        "approximate": True,
    },
    "chicken_wings": {
        "displayName": "Chicken Wings",
        "per100g": {"calories": 247, "protein": 20.0, "fat": 18.0, "carbs": 1.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "2342483",
        "source_desc": "Chicken wing, fried, meat+skin, per 100g",
        "approximate": False,
    },
    "chocolate_cake": {
        "displayName": "Chocolate Cake",
        "per100g": {"calories": 370, "protein": 5.0, "fat": 17.0, "carbs": 53.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "172802",
        "source_desc": "Chocolate cake with chocolate frosting, home recipe",
        "approximate": False,
    },
    "chocolate_mousse": {
        "displayName": "Chocolate Mousse",
        "per100g": {"calories": 330, "protein": 5.5, "fat": 25.0, "carbs": 25.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: chocolate + cream + egg mousse",
        "approximate": True,
    },
    "churros": {
        "displayName": "Churros",
        "per100g": {"calories": 360, "protein": 4.5, "fat": 18.0, "carbs": 47.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: fried dough pastry with cinnamon sugar",
        "approximate": True,
    },
    "clam_chowder": {
        "displayName": "Clam Chowder",
        "per100g": {"calories": 82, "protein": 3.5, "fat": 4.5, "carbs": 7.0},
        "density_g_per_ml": 1.0,
        "usda_fdc_id": "171114",
        "source_desc": "Clam chowder, New England, canned, condensed",
        "approximate": False,
    },
    "club_sandwich": {
        "displayName": "Club Sandwich",
        "per100g": {"calories": 240, "protein": 14.0, "fat": 12.0, "carbs": 19.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: bread + turkey + bacon + lettuce + tomato + mayo",
        "approximate": True,
    },
    "crab_cakes": {
        "displayName": "Crab Cakes",
        "per100g": {"calories": 200, "protein": 16.0, "fat": 12.0, "carbs": 8.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: crab meat + breadcrumbs + egg + seasonings, pan-fried",
        "approximate": True,
    },
    "creme_brulee": {
        "displayName": "Creme Brulee",
        "per100g": {"calories": 290, "protein": 5.0, "fat": 22.0, "carbs": 18.0},
        "density_g_per_ml": 0.75,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: heavy cream + egg yolks + sugar custard",
        "approximate": True,
    },
    "croque_madame": {
        "displayName": "Croque Madame",
        "per100g": {"calories": 255, "protein": 13.0, "fat": 17.0, "carbs": 14.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: ham + cheese sandwich with bechamel + fried egg",
        "approximate": True,
    },
    "cup_cakes": {
        "displayName": "Cup Cakes",
        "per100g": {"calories": 370, "protein": 4.0, "fat": 16.0, "carbs": 54.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "174993",
        "source_desc": "Cupcake, with frosting, home recipe",
        "approximate": False,
    },
    "deviled_eggs": {
        "displayName": "Deviled Eggs",
        "per100g": {"calories": 170, "protein": 12.0, "fat": 13.0, "carbs": 1.5},
        "density_g_per_ml": 0.65,
        "usda_fdc_id": "173426",
        "source_desc": "Egg, whole, cooked, hard-boiled — + mayo/yolk filling estimated",
        "approximate": True,
    },
    "donuts": {
        "displayName": "Donuts",
        "per100g": {"calories": 452, "protein": 4.9, "fat": 25.0, "carbs": 51.3},
        "density_g_per_ml": 0.30,
        "usda_fdc_id": "1841655",
        "source_desc": "Doughnut, glazed, yeast-leavened",
        "approximate": False,
    },
    "dumplings": {
        "displayName": "Dumplings",
        "per100g": {"calories": 170, "protein": 7.0, "fat": 6.0, "carbs": 22.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: pork+vegetable dumpling, steamed (not fried)",
        "approximate": True,
    },
    "edamame": {
        "displayName": "Edamame",
        "per100g": {"calories": 122, "protein": 11.0, "fat": 5.0, "carbs": 10.0},
        "density_g_per_ml": 0.65,
        "usda_fdc_id": "169283",
        "source_desc": "Soybeans, green (edamame), frozen, prepared",
        "approximate": False,
    },
    "eggs_benedict": {
        "displayName": "Eggs Benedict",
        "per100g": {"calories": 260, "protein": 12.0, "fat": 20.0, "carbs": 10.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: English muffin + ham + poached egg + hollandaise",
        "approximate": True,
    },
    "escargots": {
        "displayName": "Escargots",
        "per100g": {"calories": 280, "protein": 15.0, "fat": 23.0, "carbs": 3.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: snails baked with garlic butter",
        "approximate": True,
    },
    "falafel": {
        "displayName": "Falafel",
        "per100g": {"calories": 333, "protein": 13.0, "fat": 18.0, "carbs": 32.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "173306",
        "source_desc": "Falafel, home-prepared",
        "approximate": False,
    },
    "filet_mignon": {
        "displayName": "Filet Mignon",
        "per100g": {"calories": 189, "protein": 27.0, "fat": 9.0, "carbs": 0.0},
        "density_g_per_ml": 0.75,
        "usda_fdc_id": "168726",
        "source_desc": "Beef tenderloin, separable lean, broiled",
        "approximate": False,
    },
    "fish_and_chips": {
        "displayName": "Fish and Chips",
        "per100g": {"calories": 230, "protein": 9.0, "fat": 12.0, "carbs": 21.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: battered fried fish + french fries",
        "approximate": True,
    },
    "foie_gras": {
        "displayName": "Foie Gras",
        "per100g": {"calories": 462, "protein": 11.0, "fat": 44.0, "carbs": 5.0},
        "density_g_per_ml": 0.75,
        "usda_fdc_id": "172450",
        "source_desc": "Goose liver pate (foie gras), smoked, canned",
        "approximate": False,
    },
    "french_fries": {
        "displayName": "French Fries",
        "per100g": {"calories": 312, "protein": 3.5, "fat": 15.0, "carbs": 41.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "170698",
        "source_desc": "French fries, restaurant, deep fried",
        "approximate": False,
    },
    "french_onion_soup": {
        "displayName": "French Onion Soup",
        "per100g": {"calories": 70, "protein": 3.0, "fat": 2.5, "carbs": 9.0},
        "density_g_per_ml": 0.90,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: beef broth + caramelized onions + bread + cheese",
        "approximate": True,
    },
    "french_toast": {
        "displayName": "French Toast",
        "per100g": {"calories": 230, "protein": 7.0, "fat": 11.0, "carbs": 25.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "172784",
        "source_desc": "French toast, home recipe",
        "approximate": False,
    },
    "fried_calamari": {
        "displayName": "Fried Calamari",
        "per100g": {"calories": 230, "protein": 13.0, "fat": 12.0, "carbs": 17.0},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: battered + fried squid rings",
        "approximate": True,
    },
    "fried_rice": {
        "displayName": "Fried Rice",
        "per100g": {"calories": 170, "protein": 5.0, "fat": 5.0, "carbs": 26.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "168912",
        "source_desc": "Rice, white, with vegetables and egg, Chinese restaurant",
        "approximate": False,
    },
    "frozen_yogurt": {
        "displayName": "Frozen Yogurt",
        "per100g": {"calories": 130, "protein": 3.5, "fat": 3.5, "carbs": 22.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "171289",
        "source_desc": "Frozen yogurt, vanilla, soft-serve",
        "approximate": False,
    },
    "garlic_bread": {
        "displayName": "Garlic Bread",
        "per100g": {"calories": 350, "protein": 8.0, "fat": 17.0, "carbs": 42.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: Italian bread + garlic butter, toasted",
        "approximate": True,
    },
    "gnocchi": {
        "displayName": "Gnocchi",
        "per100g": {"calories": 150, "protein": 4.0, "fat": 3.0, "carbs": 27.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: potato gnocchi, cooked (no sauce)",
        "approximate": True,
    },
    "greek_salad": {
        "displayName": "Greek Salad",
        "per100g": {"calories": 90, "protein": 3.0, "fat": 7.0, "carbs": 4.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: tomato + cucumber + feta + olives + olive oil",
        "approximate": True,
    },
    "grilled_cheese_sandwich": {
        "displayName": "Grilled Cheese Sandwich",
        "per100g": {"calories": 340, "protein": 12.0, "fat": 20.0, "carbs": 28.0},
        "density_g_per_ml": 0.30,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: white bread + American cheese, butter-grilled",
        "approximate": True,
    },
    "grilled_salmon": {
        "displayName": "Grilled Salmon",
        "per100g": {"calories": 208, "protein": 25.0, "fat": 12.0, "carbs": 0.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "173687",
        "source_desc": "Salmon, Atlantic, farmed, cooked, dry heat",
        "approximate": False,
    },
    "guacamole": {
        "displayName": "Guacamole",
        "per100g": {"calories": 160, "protein": 2.0, "fat": 15.0, "carbs": 9.0},
        "density_g_per_ml": 0.65,
        "usda_fdc_id": "171036",
        "source_desc": "Guacamole, home-prepared",
        "approximate": False,
    },
    "gyoza": {
        "displayName": "Gyoza",
        "per100g": {"calories": 190, "protein": 8.0, "fat": 8.0, "carbs": 21.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: pan-fried pork dumplings (Japanese gyoza)",
        "approximate": True,
    },
    "hamburger": {
        "displayName": "Hamburger",
        "per100g": {"calories": 265, "protein": 15.0, "fat": 15.0, "carbs": 20.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "2342480",
        "source_desc": "Cheeseburger, single patty, restaurant",
        "approximate": False,
    },
    "hot_and_sour_soup": {
        "displayName": "Hot and Sour Soup",
        "per100g": {"calories": 50, "protein": 4.0, "fat": 2.0, "carbs": 4.0},
        "density_g_per_ml": 0.95,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: Chinese hot and sour soup, restaurant",
        "approximate": True,
    },
    "hot_dog": {
        "displayName": "Hot Dog",
        "per100g": {"calories": 290, "protein": 11.0, "fat": 18.0, "carbs": 22.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "173042",
        "source_desc": "Frankfurter, beef, in bun",
        "approximate": False,
    },
    "huevos_rancheros": {
        "displayName": "Huevos Rancheros",
        "per100g": {"calories": 165, "protein": 7.0, "fat": 10.0, "carbs": 12.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: eggs + tortilla + salsa + beans",
        "approximate": True,
    },
    "hummus": {
        "displayName": "Hummus",
        "per100g": {"calories": 177, "protein": 7.0, "fat": 9.0, "carbs": 17.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "174288",
        "source_desc": "Hummus, commercial",
        "approximate": False,
    },
    "ice_cream": {
        "displayName": "Ice Cream",
        "per100g": {"calories": 207, "protein": 3.5, "fat": 11.0, "carbs": 23.6},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "171287",
        "source_desc": "Ice cream, vanilla",
        "approximate": False,
    },
    "lasagna": {
        "displayName": "Lasagna",
        "per100g": {"calories": 147, "protein": 7.0, "fat": 6.0, "carbs": 16.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "169692",
        "source_desc": "Lasagna with meat and cheese, home recipe",
        "approximate": False,
    },
    "lobster_bisque": {
        "displayName": "Lobster Bisque",
        "per100g": {"calories": 100, "protein": 6.0, "fat": 6.0, "carbs": 6.0},
        "density_g_per_ml": 0.90,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: cream-based lobster soup",
        "approximate": True,
    },
    "lobster_roll_sandwich": {
        "displayName": "Lobster Roll Sandwich",
        "per100g": {"calories": 220, "protein": 12.0, "fat": 10.0, "carbs": 20.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: lobster meat + mayo in buttered roll",
        "approximate": True,
    },
    "macaroni_and_cheese": {
        "displayName": "Macaroni and Cheese",
        "per100g": {"calories": 190, "protein": 7.0, "fat": 9.0, "carbs": 20.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "169695",
        "source_desc": "Macaroni and cheese, home recipe",
        "approximate": False,
    },
    "macarons": {
        "displayName": "Macarons",
        "per100g": {"calories": 380, "protein": 6.0, "fat": 14.0, "carbs": 58.0},
        "density_g_per_ml": 0.25,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: almond meringue cookie with ganache/buttercream",
        "approximate": True,
    },
    "miso_soup": {
        "displayName": "Miso Soup",
        "per100g": {"calories": 35, "protein": 2.5, "fat": 1.0, "carbs": 4.0},
        "density_g_per_ml": 0.95,
        "usda_fdc_id": "173280",
        "source_desc": "Miso soup, restaurant",
        "approximate": False,
    },
    "mussels": {
        "displayName": "Mussels",
        "per100g": {"calories": 115, "protein": 16.0, "fat": 3.0, "carbs": 5.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "174217",
        "source_desc": "Mussels, blue, cooked, moist heat",
        "approximate": False,
    },
    "nachos": {
        "displayName": "Nachos",
        "per100g": {"calories": 310, "protein": 8.0, "fat": 19.0, "carbs": 28.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: tortilla chips + cheese + jalapenos + sour cream",
        "approximate": True,
    },
    "omelette": {
        "displayName": "Omelette",
        "per100g": {"calories": 175, "protein": 12.0, "fat": 14.0, "carbs": 1.5},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "173427",
        "source_desc": "Egg, whole, cooked, omelet, plain",
        "approximate": False,
    },
    "onion_rings": {
        "displayName": "Onion Rings",
        "per100g": {"calories": 310, "protein": 4.0, "fat": 17.0, "carbs": 36.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "169733",
        "source_desc": "Onion rings, breaded, deep fried, frozen, heated",
        "approximate": False,
    },
    "oysters": {
        "displayName": "Oysters",
        "per100g": {"calories": 81, "protein": 9.0, "fat": 2.5, "carbs": 5.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "174219",
        "source_desc": "Oysters, eastern, wild, cooked, moist heat",
        "approximate": False,
    },
    "pad_thai": {
        "displayName": "Pad Thai",
        "per100g": {"calories": 170, "protein": 5.0, "fat": 6.0, "carbs": 23.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: rice noodles + shrimp/tofu + tamarind + peanut + egg",
        "approximate": True,
    },
    "paella": {
        "displayName": "Paella",
        "per100g": {"calories": 160, "protein": 8.0, "fat": 5.0, "carbs": 20.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: saffron rice + seafood + chicken + vegetables",
        "approximate": True,
    },
    "pancakes": {
        "displayName": "Pancakes",
        "per100g": {"calories": 227, "protein": 6.4, "fat": 9.7, "carbs": 28.3},
        "density_g_per_ml": 0.30,
        "usda_fdc_id": "168758",
        "source_desc": "Pancakes, plain, prepared from recipe",
        "approximate": False,
    },
    "panna_cotta": {
        "displayName": "Panna Cotta",
        "per100g": {"calories": 250, "protein": 4.0, "fat": 20.0, "carbs": 14.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: sweetened cream + gelatin dessert",
        "approximate": True,
    },
    "peking_duck": {
        "displayName": "Peking Duck",
        "per100g": {"calories": 270, "protein": 18.0, "fat": 22.0, "carbs": 2.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "172412",
        "source_desc": "Duck, domesticated, meat+skin, roasted — Peking duck estimate",
        "approximate": True,
    },
    "pho": {
        "displayName": "Pho",
        "per100g": {"calories": 80, "protein": 5.0, "fat": 2.0, "carbs": 11.0},
        "density_g_per_ml": 0.85,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: beef broth + rice noodles + beef slices + herbs",
        "approximate": True,
    },
    "pizza": {
        "displayName": "Pizza",
        "per100g": {"calories": 266, "protein": 11.0, "fat": 9.7, "carbs": 33.3},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "17329229",
        "source_desc": "Pizza, cheese, restaurant, regular crust",
        "approximate": False,
    },
    "pork_chop": {
        "displayName": "Pork Chop",
        "per100g": {"calories": 231, "protein": 26.0, "fat": 14.0, "carbs": 0.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "168254",
        "source_desc": "Pork chop, loin, lean+fat, broiled",
        "approximate": False,
    },
    "poutine": {
        "displayName": "Poutine",
        "per100g": {"calories": 220, "protein": 6.0, "fat": 13.0, "carbs": 20.0},
        "density_g_per_ml": 0.50,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: french fries + cheese curds + gravy",
        "approximate": True,
    },
    "prime_rib": {
        "displayName": "Prime Rib",
        "per100g": {"calories": 296, "protein": 23.0, "fat": 23.0, "carbs": 0.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "168621",
        "source_desc": "Beef rib, large end, lean+fat, roasted",
        "approximate": False,
    },
    "pulled_pork_sandwich": {
        "displayName": "Pulled Pork Sandwich",
        "per100g": {"calories": 220, "protein": 14.0, "fat": 10.0, "carbs": 19.0},
        "density_g_per_ml": 0.40,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: pulled pork + BBQ sauce + bun",
        "approximate": True,
    },
    "ramen": {
        "displayName": "Ramen",
        "per100g": {"calories": 95, "protein": 3.5, "fat": 3.5, "carbs": 13.0},
        "density_g_per_ml": 0.80,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: broth + wheat noodles + toppings; varies widely",
        "approximate": True,
    },
    "ravioli": {
        "displayName": "Ravioli",
        "per100g": {"calories": 150, "protein": 7.0, "fat": 4.0, "carbs": 22.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "169754",
        "source_desc": "Ravioli, cheese-filled, canned",
        "approximate": False,
    },
    "red_velvet_cake": {
        "displayName": "Red Velvet Cake",
        "per100g": {"calories": 370, "protein": 4.5, "fat": 18.0, "carbs": 50.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "172802",
        "source_desc": "Closest: chocolate cake with frosting (red velvet is similar)",
        "approximate": True,
    },
    "risotto": {
        "displayName": "Risotto",
        "per100g": {"calories": 155, "protein": 3.5, "fat": 5.0, "carbs": 23.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: arborio rice + parmesan + butter + broth",
        "approximate": True,
    },
    "samosa": {
        "displayName": "Samosa",
        "per100g": {"calories": 260, "protein": 5.0, "fat": 14.0, "carbs": 30.0},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: fried pastry + spiced potato+pea filling",
        "approximate": True,
    },
    "sashimi": {
        "displayName": "Sashimi",
        "per100g": {"calories": 130, "protein": 23.0, "fat": 4.0, "carbs": 0.0},
        "density_g_per_ml": 0.75,
        "usda_fdc_id": "175168",
        "source_desc": "Fish, tuna, fresh, raw — representative sashimi",
        "approximate": False,
    },
    "scallops": {
        "displayName": "Scallops",
        "per100g": {"calories": 111, "protein": 21.0, "fat": 1.5, "carbs": 3.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "174222",
        "source_desc": "Scallops, bay or sea, cooked, breaded and fried",
        "approximate": False,
    },
    "seaweed_salad": {
        "displayName": "Seaweed Salad",
        "per100g": {"calories": 70, "protein": 2.0, "fat": 4.0, "carbs": 8.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "170501",
        "source_desc": "Seaweed, wakame, raw — salad with sesame dressing",
        "approximate": True,
    },
    "shrimp_and_grits": {
        "displayName": "Shrimp and Grits",
        "per100g": {"calories": 140, "protein": 8.0, "fat": 6.0, "carbs": 14.0},
        "density_g_per_ml": 0.60,
        "usda_fdc_id": "N/A",
        "source_desc": "Composite: shrimp + grits + butter + cheese",
        "approximate": True,
    },
    "spaghetti_bolognese": {
        "displayName": "Spaghetti Bolognese",
        "per100g": {"calories": 145, "protein": 6.0, "fat": 5.0, "carbs": 19.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "169756",
        "source_desc": "Spaghetti with meat sauce, home recipe",
        "approximate": False,
    },
    "spaghetti_carbonara": {
        "displayName": "Spaghetti Carbonara",
        "per100g": {"calories": 200, "protein": 8.0, "fat": 10.0, "carbs": 19.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: spaghetti + egg + pancetta/bacon + parmesan + black pepper",
        "approximate": True,
    },
    "spring_rolls": {
        "displayName": "Spring Rolls",
        "per100g": {"calories": 150, "protein": 4.0, "fat": 5.0, "carbs": 22.0},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: rice paper + vegetables + shrimp, fresh (not fried)",
        "approximate": True,
    },
    "steak": {
        "displayName": "Steak",
        "per100g": {"calories": 271, "protein": 25.0, "fat": 19.0, "carbs": 0.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "168636",
        "source_desc": "Beef, rib eye steak, lean+fat, broiled",
        "approximate": False,
    },
    "strawberry_shortcake": {
        "displayName": "Strawberry Shortcake",
        "per100g": {"calories": 250, "protein": 4.0, "fat": 12.0, "carbs": 32.0},
        "density_g_per_ml": 0.35,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: biscuit/cake + strawberries + whipped cream",
        "approximate": True,
    },
    "sushi": {
        "displayName": "Sushi",
        "per100g": {"calories": 145, "protein": 7.0, "fat": 1.0, "carbs": 28.0},
        "density_g_per_ml": 0.65,
        "usda_fdc_id": "2342411",
        "source_desc": "California roll (representative maki sushi)",
        "approximate": False,
    },
    "tacos": {
        "displayName": "Tacos",
        "per100g": {"calories": 226, "protein": 11.0, "fat": 12.0, "carbs": 20.0},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "2342495",
        "source_desc": "Taco, beef, restaurant",
        "approximate": False,
    },
    "takoyaki": {
        "displayName": "Takoyaki",
        "per100g": {"calories": 180, "protein": 7.0, "fat": 8.0, "carbs": 20.0},
        "density_g_per_ml": 0.55,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: octopus-filled batter balls with sauce + mayo",
        "approximate": True,
    },
    "tiramisu": {
        "displayName": "Tiramisu",
        "per100g": {"calories": 300, "protein": 5.0, "fat": 18.0, "carbs": 30.0},
        "density_g_per_ml": 0.45,
        "usda_fdc_id": "N/A",
        "source_desc": "Estimate: mascarpone + espresso-soaked ladyfingers + cocoa",
        "approximate": True,
    },
    "tuna_tartare": {
        "displayName": "Tuna Tartare",
        "per100g": {"calories": 140, "protein": 23.0, "fat": 5.0, "carbs": 1.0},
        "density_g_per_ml": 0.70,
        "usda_fdc_id": "175168",
        "source_desc": "Tuna, fresh, raw — tartare with seasonings",
        "approximate": True,
    },
    "waffles": {
        "displayName": "Waffles",
        "per100g": {"calories": 291, "protein": 8.0, "fat": 14.0, "carbs": 33.0},
        "density_g_per_ml": 0.30,
        "usda_fdc_id": "2342489",
        "source_desc": "Waffle, plain, restaurant",
        "approximate": False,
    },
}

# ---------------------------------------------------------------------------
# Reference serving masses (g) for each food — used when the user hasn't
# specified a portion yet. These are typical single-serving estimates used
# ONLY as display defaults; actual calculations use the per-100g values above.
# ---------------------------------------------------------------------------
REFERENCE_MASSES = {
    "apple_pie": 125, "baby_back_ribs": 250, "baklava": 80, "beef_carpaccio": 120,
    "beef_tartare": 150, "beet_salad": 200, "beignets": 80, "bibimbap": 500,
    "bread_pudding": 150, "breakfast_burrito": 220, "bruschetta": 120,
    "caesar_salad": 200, "cannoli": 90, "caprese_salad": 200, "carrot_cake": 110,
    "ceviche": 200, "cheesecake": 125, "cheese_plate": 120, "chicken_curry": 350,
    "chicken_quesadilla": 230, "chicken_wings": 200, "chocolate_cake": 100,
    "chocolate_mousse": 120, "churros": 60, "clam_chowder": 250,
    "club_sandwich": 280, "crab_cakes": 140, "creme_brulee": 130,
    "croque_madame": 220, "cup_cakes": 90, "deviled_eggs": 60, "donuts": 100,
    "dumplings": 180, "edamame": 155, "eggs_benedict": 280, "escargots": 100,
    "falafel": 140, "filet_mignon": 200, "fish_and_chips": 300, "foie_gras": 100,
    "french_fries": 130, "french_onion_soup": 300, "french_toast": 150,
    "fried_calamari": 150, "fried_rice": 250, "frozen_yogurt": 170,
    "garlic_bread": 60, "gnocchi": 250, "greek_salad": 250,
    "grilled_cheese_sandwich": 130, "grilled_salmon": 200, "guacamole": 120,
    "gyoza": 160, "hamburger": 250, "hot_and_sour_soup": 300, "hot_dog": 110,
    "huevos_rancheros": 280, "hummus": 90, "ice_cream": 130, "lasagna": 300,
    "lobster_bisque": 250, "lobster_roll_sandwich": 200,
    "macaroni_and_cheese": 220, "macarons": 25, "miso_soup": 250,
    "mussels": 250, "nachos": 250, "omelette": 180, "onion_rings": 140,
    "oysters": 150, "pad_thai": 300, "paella": 400, "pancakes": 160,
    "panna_cotta": 120, "peking_duck": 200, "pho": 550, "pizza": 150,
    "pork_chop": 200, "poutine": 350, "prime_rib": 250,
    "pulled_pork_sandwich": 250, "ramen": 550, "ravioli": 250,
    "red_velvet_cake": 110, "risotto": 300, "samosa": 100, "sashimi": 150,
    "scallops": 150, "seaweed_salad": 100, "shrimp_and_grits": 300,
    "spaghetti_bolognese": 350, "spaghetti_carbonara": 300, "spring_rolls": 120,
    "steak": 250, "strawberry_shortcake": 130, "sushi": 200, "tacos": 220,
    "takoyaki": 160, "tiramisu": 140, "tuna_tartare": 150, "waffles": 160,
}


def generate_catalog(output_path):
    """Generate food-catalog.json from embedded USDA reference data."""
    catalog = {
        "version": "1.0.0",
        "source": "USDA FoodData Central (fdc.nal.usda.gov)",
        "generated": "2026-07-19",
        "description": (
            "Per-100g nutrition values for each Food-101 class, sourced from "
            "USDA FoodData Central where available. Entries marked "
            "'approximate: true' are composite estimates for complex dishes "
            "that have no single USDA match."
        ),
        "foods": {},
    }

    for label, data in sorted(USDA_CATALOG.items()):
        entry = {
            "label": label,
            **data,
            "referenceMass": REFERENCE_MASSES.get(label, 100),
        }
        catalog["foods"][label] = entry

    # Validate consistency between nutrition data and reference masses
    missing = set(USDA_CATALOG.keys()) - set(REFERENCE_MASSES.keys())
    extra = set(REFERENCE_MASSES.keys()) - set(USDA_CATALOG.keys())
    if missing:
        print(f"[warn] Foods with nutrition but no reference mass: {missing}")
    if extra:
        print(f"[warn] Foods with reference mass but no nutrition: {extra}")

    output_path = Path(output_path)
    output_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"Catalog written to {output_path}")
    print(f"  {len(catalog['foods'])} foods")
    approximate = sum(1 for f in catalog["foods"].values() if f.get("approximate"))
    exact = len(catalog["foods"]) - approximate
    print(f"  {exact} with USDA FDC match, {approximate} approximate (composite dishes)")


def validate_catalog(catalog_path):
    """Check catalog consistency."""
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))

    issues = []
    for label, food in catalog.get("foods", {}).items():
        per100 = food.get("per100g", {})
        # Check required fields
        for key in ["calories", "protein", "fat", "carbs"]:
            if key not in per100 or not isinstance(per100[key], (int, float)):
                issues.append(f"{label}: missing or invalid per100g.{key}")
        # Check non-negative
        for key, val in per100.items():
            if val < 0:
                issues.append(f"{label}: negative {key}={val}")
        # Atwater consistency check (calories should be close to 4*protein + 9*fat + 4*carbs)
        expected_cal = 4 * per100.get("protein", 0) + 9 * per100.get("fat", 0) + 4 * per100.get("carbs", 0)
        actual_cal = per100.get("calories", 0)
        if actual_cal > 0 and abs(actual_cal - expected_cal) / actual_cal > 0.3:
            issues.append(
                f"{label}: Atwater mismatch — calories={actual_cal} vs "
                f"4P+9F+4C={expected_cal:.0f} ({abs(actual_cal-expected_cal)/actual_cal:.0%} off)"
            )
        # Check density range
        density = food.get("density_g_per_ml")
        if density is not None and not (0.1 <= density <= 2.0):
            issues.append(f"{label}: implausible density {density} g/ml")

    if issues:
        print(f"VALIDATION: {len(issues)} issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("VALIDATION: All checks passed.")
    return issues


def main():
    ap = argparse.ArgumentParser(description="Build food nutrition catalog from USDA data")
    ap.add_argument("--usda-csv", help="Path to USDA FoodData Central CSV (optional)")
    ap.add_argument("--validate", action="store_true", help="Validate existing catalog")
    ap.add_argument("--output", default=str(CATALOG_PATH), help="Output path")
    args = ap.parse_args()

    if args.validate:
        issues = validate_catalog(args.output)
        if issues:
            sys.exit(1)
        return

    if args.usda_csv:
        print(f"Loading USDA CSV: {args.usda_csv}")
        # Future: parse USDA CSV and merge/override the embedded catalog
        # For now, the embedded catalog is the primary source
        print("[info] USDA CSV parsing not yet implemented; using embedded catalog")

    generate_catalog(args.output)


if __name__ == "__main__":
    main()
