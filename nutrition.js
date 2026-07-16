/*
 * Food Tracker - Food-101 label list + nutrition lookup table.
 *
 * The browser model (onnx-community/swin-finetuned-food101-ONNX) is an image
 * *classifier*: it predicts one of the 101 Food-101 dish categories. It does
 * NOT directly regress calories. So we map the predicted dish to typical
 * per-serving nutrition values below.
 *
 * Values are approximate, per a single typical serving:
 *   calories (kcal), mass (g), protein (g), fat (g), carbs (g)
 * They are meant for demonstration/estimation, not medical/dietary precision.
 *
 * LABELS order MUST match the model's output logits index (config.json
 * id2label from the checkpoint). Do not reorder.
 */
(function (global) {
  const FT = (global.FT = global.FT || {});

  // Food-101 classes in the exact order of the model's 101 output logits.
  const LABELS = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict',
    'escargots', 'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras',
    'french_fries', 'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna',
    'lobster_bisque', 'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters',
    'pad_thai', 'paella', 'pancakes', 'panna_cotta', 'peking_duck',
    'pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib',
    'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 'strawberry_shortcake',
    'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare', 'waffles'
  ];

  // Per-serving nutrition: [calories(kcal), mass(g), protein(g), fat(g), carbs(g)]
  const NUTRITION = {
    apple_pie: [296, 125, 2, 14, 43],
    baby_back_ribs: [500, 250, 35, 38, 5],
    baklava: [334, 80, 5, 20, 37],
    beef_carpaccio: [190, 120, 22, 11, 1],
    beef_tartare: [220, 150, 24, 12, 2],
    beet_salad: [150, 200, 4, 7, 20],
    beignets: [300, 80, 5, 16, 35],
    bibimbap: [550, 500, 20, 15, 80],
    bread_pudding: [320, 150, 7, 12, 45],
    breakfast_burrito: [430, 220, 18, 22, 40],
    bruschetta: [190, 120, 5, 8, 25],
    caesar_salad: [330, 200, 9, 27, 12],
    cannoli: [280, 90, 6, 16, 30],
    caprese_salad: [280, 200, 14, 22, 8],
    carrot_cake: [415, 110, 4, 22, 52],
    ceviche: [180, 200, 24, 5, 9],
    cheesecake: [400, 125, 7, 28, 32],
    cheese_plate: [430, 120, 25, 35, 3],
    chicken_curry: [430, 350, 30, 22, 25],
    chicken_quesadilla: [510, 230, 27, 27, 40],
    chicken_wings: [430, 200, 35, 30, 5],
    chocolate_cake: [370, 100, 5, 18, 50],
    chocolate_mousse: [355, 120, 6, 25, 30],
    churros: [230, 60, 3, 12, 28],
    clam_chowder: [200, 250, 9, 10, 18],
    club_sandwich: [590, 280, 33, 30, 45],
    crab_cakes: [290, 140, 18, 18, 12],
    creme_brulee: [340, 130, 5, 25, 25],
    croque_madame: [510, 220, 27, 30, 30],
    cup_cakes: [305, 90, 3, 14, 43],
    deviled_eggs: [140, 60, 6, 12, 1],
    donuts: [452, 100, 5, 25, 51],
    dumplings: [280, 180, 11, 10, 35],
    edamame: [190, 155, 17, 8, 15],
    eggs_benedict: [730, 280, 30, 55, 25],
    escargots: [250, 100, 14, 20, 3],
    falafel: [330, 140, 13, 18, 32],
    filet_mignon: [350, 200, 46, 18, 0],
    fish_and_chips: [760, 300, 32, 42, 65],
    foie_gras: [460, 100, 11, 44, 5],
    french_fries: [365, 130, 4, 17, 48],
    french_onion_soup: [370, 300, 15, 22, 28],
    french_toast: [350, 150, 11, 14, 45],
    fried_calamari: [300, 150, 18, 15, 25],
    fried_rice: [440, 250, 12, 14, 65],
    frozen_yogurt: [220, 170, 6, 6, 38],
    garlic_bread: [200, 60, 5, 9, 25],
    gnocchi: [370, 250, 9, 8, 65],
    greek_salad: [230, 250, 6, 18, 12],
    grilled_cheese_sandwich: [400, 130, 15, 24, 32],
    grilled_salmon: [370, 200, 40, 22, 0],
    guacamole: [230, 120, 3, 21, 12],
    gyoza: [260, 160, 10, 12, 28],
    hamburger: [540, 250, 30, 27, 42],
    hot_and_sour_soup: [160, 300, 9, 7, 15],
    hot_dog: [290, 110, 11, 18, 22],
    huevos_rancheros: [460, 280, 20, 26, 38],
    hummus: [180, 90, 5, 12, 15],
    ice_cream: [275, 130, 5, 15, 32],
    lasagna: [480, 300, 25, 24, 40],
    lobster_bisque: [320, 250, 14, 22, 15],
    lobster_roll_sandwich: [440, 200, 22, 24, 35],
    macaroni_and_cheese: [400, 220, 15, 20, 42],
    macarons: [90, 25, 2, 4, 12],
    miso_soup: [70, 250, 5, 3, 7],
    mussels: [290, 250, 30, 8, 20],
    nachos: [560, 250, 16, 33, 52],
    omelette: [330, 180, 22, 24, 3],
    onion_rings: [410, 140, 6, 24, 45],
    oysters: [130, 150, 14, 4, 8],
    pad_thai: [560, 300, 20, 20, 75],
    paella: [630, 400, 32, 22, 70],
    pancakes: [350, 160, 8, 12, 52],
    panna_cotta: [290, 120, 4, 20, 24],
    peking_duck: [420, 200, 24, 32, 8],
    pho: [480, 550, 30, 12, 65],
    pizza: [285, 107, 12, 10, 36],
    pork_chop: [360, 200, 40, 22, 0],
    poutine: [740, 350, 20, 42, 72],
    prime_rib: [640, 250, 42, 52, 0],
    pulled_pork_sandwich: [530, 250, 30, 22, 50],
    ramen: [500, 550, 22, 18, 65],
    ravioli: [390, 250, 15, 14, 50],
    red_velvet_cake: [400, 110, 4, 20, 52],
    risotto: [430, 300, 10, 16, 60],
    samosa: [260, 100, 5, 14, 30],
    sashimi: [200, 150, 32, 7, 1],
    scallops: [200, 150, 24, 8, 6],
    seaweed_salad: [110, 100, 2, 6, 12],
    shrimp_and_grits: [480, 300, 26, 24, 38],
    spaghetti_bolognese: [560, 350, 24, 20, 68],
    spaghetti_carbonara: [620, 300, 22, 30, 65],
    spring_rolls: [200, 120, 6, 10, 22],
    steak: [500, 250, 46, 34, 0],
    strawberry_shortcake: [340, 130, 4, 16, 46],
    sushi: [350, 200, 14, 6, 60],
    tacos: [430, 220, 20, 22, 38],
    takoyaki: [320, 160, 12, 14, 36],
    tiramisu: [420, 140, 7, 28, 36],
    tuna_tartare: [220, 150, 26, 11, 3],
    waffles: [410, 160, 9, 20, 48]
  };

  // Turn a class index (from argmax of logits) into a nutrition object.
  function nutritionForIndex(idx) {
    const label = LABELS[idx] || 'unknown';
    return nutritionForLabel(label);
  }

  function nutritionForLabel(label) {
    const v = NUTRITION[label] || [0, 0, 0, 0, 0];
    return {
      label: label,
      calories: v[0],
      mass: v[1],
      protein: v[2],
      fat: v[3],
      carbs: v[4]
    };
  }

  // Pretty display name, e.g. "chicken_curry" -> "Chicken Curry".
  function prettyLabel(label) {
    return String(label || '')
      .split('_')
      .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
      .join(' ');
  }

  FT.nutrition = {
    LABELS,
    NUTRITION,
    nutritionForIndex,
    nutritionForLabel,
    prettyLabel
  };
})(typeof window !== 'undefined' ? window : globalThis);
