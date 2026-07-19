/*
 * Food Tracker - Food-101 label list + USDA-sourced nutrition catalog.
 *
 * The browser model (onnx-community/swin-finetuned-food101-ONNX) is an image
 * *classifier*: it predicts one of the 101 Food-101 dish categories. It does
 * NOT directly regress calories. So we map the predicted dish to nutrition
 * values from a versioned, USDA FoodData Central-sourced catalog.
 *
 * All values are per 100 g. The user adjusts the portion in grams, and the
 * app scales linearly:  nutrients = grams × per100g / 100.
 *
 * Values are estimates — NOT medical/dietary precision.
 * See food-catalog.json for provenance, USDA FDC IDs, and approximate flags.
 *
 * LABELS order MUST match the model's output logits index (config.json
 * id2label from the checkpoint). Do not reorder.
 */

// Increment when the catalog values or structure change.
const CATALOG_VERSION = '1.0.0';

(function (global) {
  const FT = (global.FT = global.FT || {});

  // Food-101 classes in the exact order of the model's 101 output logits.
  // This array MUST stay in sync with the Swin Food-101 checkpoint's id2label.
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

  // Per-100g nutrition: [calories(kcal), referenceMass(100g), protein(g), fat(g), carbs(g)]
  // Sourced from USDA FoodData Central (see food-catalog.json for FDC IDs).
  // Entries marked `appx: true` are composite estimates for complex dishes with
  // no single USDA match — they are directionally reasonable but not exact.
  //
  // All values are per 100 g so the app computes:
  //   nutrients = user_grams × per_100g_value / 100
  const NUTRITION = {
    apple_pie:               [265, 100, 2.4, 12.5, 37.1],
    baby_back_ribs:           [297, 100, 22, 23, 0],
    baklava:                  [430, 100, 7, 24, 50],
    beef_carpaccio:           [170, 100, 21, 9, 0.5],
    beef_tartare:             [180, 100, 20, 10, 1],
    beet_salad:               [82, 100, 2.3, 4, 10],
    beignets:                 [380, 100, 6, 18, 48],
    bibimbap:                 [130, 100, 5.5, 4, 18],
    bread_pudding:            [210, 100, 5.5, 8, 30],
    breakfast_burrito:        [230, 100, 10, 12, 20],
    bruschetta:               [175, 100, 4, 7, 24],
    caesar_salad:             [157, 100, 5.2, 12.9, 5.8],
    cannoli:                  [350, 100, 8, 20, 35],
    caprese_salad:            [170, 100, 9, 13, 4],
    carrot_cake:              [390, 100, 4, 19, 50],
    ceviche:                  [85, 100, 15, 1.5, 3],
    cheesecake:               [321, 100, 5.5, 22.5, 25.5],
    cheese_plate:             [380, 100, 23, 32, 2],
    chicken_curry:            [140, 100, 10, 8, 7],
    chicken_quesadilla:       [270, 100, 14, 15, 20],
    chicken_wings:            [247, 100, 20, 18, 1],
    chocolate_cake:           [370, 100, 5, 17, 53],
    chocolate_mousse:         [330, 100, 5.5, 25, 25],
    churros:                  [360, 100, 4.5, 18, 47],
    clam_chowder:             [82, 100, 3.5, 4.5, 7],
    club_sandwich:            [240, 100, 14, 12, 19],
    crab_cakes:               [200, 100, 16, 12, 8],
    creme_brulee:             [290, 100, 5, 22, 18],
    croque_madame:            [255, 100, 13, 17, 14],
    cup_cakes:                [370, 100, 4, 16, 54],
    deviled_eggs:             [170, 100, 12, 13, 1.5],
    donuts:                   [452, 100, 4.9, 25, 51.3],
    dumplings:                [170, 100, 7, 6, 22],
    edamame:                  [122, 100, 11, 5, 10],
    eggs_benedict:            [260, 100, 12, 20, 10],
    escargots:                [280, 100, 15, 23, 3],
    falafel:                  [333, 100, 13, 18, 32],
    filet_mignon:             [189, 100, 27, 9, 0],
    fish_and_chips:           [230, 100, 9, 12, 21],
    foie_gras:                [462, 100, 11, 44, 5],
    french_fries:             [312, 100, 3.5, 15, 41],
    french_onion_soup:        [70, 100, 3, 2.5, 9],
    french_toast:             [230, 100, 7, 11, 25],
    fried_calamari:           [230, 100, 13, 12, 17],
    fried_rice:               [170, 100, 5, 5, 26],
    frozen_yogurt:            [130, 100, 3.5, 3.5, 22],
    garlic_bread:             [350, 100, 8, 17, 42],
    gnocchi:                  [150, 100, 4, 3, 27],
    greek_salad:              [90, 100, 3, 7, 4],
    grilled_cheese_sandwich:  [340, 100, 12, 20, 28],
    grilled_salmon:           [208, 100, 25, 12, 0],
    guacamole:                [160, 100, 2, 15, 9],
    gyoza:                    [190, 100, 8, 8, 21],
    hamburger:                [265, 100, 15, 15, 20],
    hot_and_sour_soup:        [50, 100, 4, 2, 4],
    hot_dog:                  [290, 100, 11, 18, 22],
    huevos_rancheros:         [165, 100, 7, 10, 12],
    hummus:                   [177, 100, 7, 9, 17],
    ice_cream:                [207, 100, 3.5, 11, 23.6],
    lasagna:                  [147, 100, 7, 6, 16],
    lobster_bisque:           [100, 100, 6, 6, 6],
    lobster_roll_sandwich:    [220, 100, 12, 10, 20],
    macaroni_and_cheese:      [190, 100, 7, 9, 20],
    macarons:                 [380, 100, 6, 14, 58],
    miso_soup:                [35, 100, 2.5, 1, 4],
    mussels:                  [115, 100, 16, 3, 5],
    nachos:                   [310, 100, 8, 19, 28],
    omelette:                 [175, 100, 12, 14, 1.5],
    onion_rings:              [310, 100, 4, 17, 36],
    oysters:                  [81, 100, 9, 2.5, 5],
    pad_thai:                 [170, 100, 5, 6, 23],
    paella:                   [160, 100, 8, 5, 20],
    pancakes:                 [227, 100, 6.4, 9.7, 28.3],
    panna_cotta:              [250, 100, 4, 20, 14],
    peking_duck:              [270, 100, 18, 22, 2],
    pho:                      [80, 100, 5, 2, 11],
    pizza:                    [266, 100, 11, 9.7, 33.3],
    pork_chop:                [231, 100, 26, 14, 0],
    poutine:                  [220, 100, 6, 13, 20],
    prime_rib:                [296, 100, 23, 23, 0],
    pulled_pork_sandwich:     [220, 100, 14, 10, 19],
    ramen:                    [95, 100, 3.5, 3.5, 13],
    ravioli:                  [150, 100, 7, 4, 22],
    red_velvet_cake:          [370, 100, 4.5, 18, 50],
    risotto:                  [155, 100, 3.5, 5, 23],
    samosa:                   [260, 100, 5, 14, 30],
    sashimi:                  [130, 100, 23, 4, 0],
    scallops:                 [111, 100, 21, 1.5, 3],
    seaweed_salad:            [70, 100, 2, 4, 8],
    shrimp_and_grits:         [140, 100, 8, 6, 14],
    spaghetti_bolognese:      [145, 100, 6, 5, 19],
    spaghetti_carbonara:      [200, 100, 8, 10, 19],
    spring_rolls:             [150, 100, 4, 5, 22],
    steak:                    [271, 100, 25, 19, 0],
    strawberry_shortcake:     [250, 100, 4, 12, 32],
    sushi:                    [145, 100, 7, 1, 28],
    tacos:                    [226, 100, 11, 12, 20],
    takoyaki:                 [180, 100, 7, 8, 20],
    tiramisu:                 [300, 100, 5, 18, 30],
    tuna_tartare:             [140, 100, 23, 5, 1],
    waffles:                  [291, 100, 8, 14, 33]
  };

  // Turn a class index (from argmax of logits) into a nutrition object.
  function nutritionForIndex(idx) {
    const label = LABELS[idx] || 'unknown';
    return nutritionForLabel(label);
  }

  // Returns per-100g nutrition for a Food-101 label.
  function nutritionForLabel(label) {
    const v = NUTRITION[label] || [0, 100, 0, 0, 0];
    return {
      label: label,
      calories: v[0],
      mass: v[1],        // always 100 (per-100g reference)
      protein: v[2],
      fat: v[3],
      carbs: v[4]
    };
  }

  // Scales per-100g nutrition to a user-selected gram amount.
  // nutrients = grams × per_100g_value / 100
  function nutritionForLabelAndMass(label, grams) {
    const base = nutritionForLabel(label);
    const g = Number(grams);
    if (!Number.isFinite(g) || g <= 0) {
      throw new Error('Portion must be greater than 0 grams');
    }
    // base.mass is always 100 (per-100g reference)
    if (base.mass <= 0) {
      throw new Error('Reference mass must be greater than 0');
    }
    const factor = g / base.mass; // g / 100
    return {
      label: base.label,
      calories: base.calories * factor,
      mass: g,
      protein: base.protein * factor,
      fat: base.fat * factor,
      carbs: base.carbs * factor
    };
  }

  // Pretty display name, e.g. "chicken_curry" -> "Chicken Curry".
  function prettyLabel(label) {
    return String(label || '')
      .split('_')
      .map(function (w) { return w ? w[0].toUpperCase() + w.slice(1) : w; })
      .join(' ');
  }

  FT.nutrition = {
    CATALOG_VERSION: CATALOG_VERSION,
    LABELS: LABELS,
    NUTRITION: NUTRITION,
    nutritionForIndex: nutritionForIndex,
    nutritionForLabel: nutritionForLabel,
    nutritionForLabelAndMass: nutritionForLabelAndMass,
    prettyLabel: prettyLabel
  };
})(typeof window !== 'undefined' ? window : globalThis);
