/**
 * ESC-50 Class Labels
 * ===================
 * The 50 environmental sound classes from the ESC-50 dataset,
 * ordered by their integer target index (0–49).
 *
 * Source: https://github.com/karolpiczak/ESC-50
 *
 * Category breakdown:
 *   0–9   Animals
 *   10–19 Natural soundscapes & water sounds
 *   20–29 Human (non-speech)
 *   30–39 Interior/domestic sounds
 *   40–49 Exterior/urban noises
 */

export const ESC50_LABELS: readonly string[] = [
  // Animals (0–9)
  "Dog",
  "Rooster",
  "Pig",
  "Cow",
  "Frog",
  "Cat",
  "Hen",
  "Insects (flying)",
  "Sheep",
  "Crow",

  // Natural soundscapes & water (10–19)
  "Rain",
  "Sea waves",
  "Crackling fire",
  "Crickets",
  "Chirping birds",
  "Water drops",
  "Wind",
  "Pouring water",
  "Toilet flush",
  "Thunderstorm",

  // Human (non-speech) (20–29)
  "Crying baby",
  "Sneezing",
  "Clapping",
  "Breathing",
  "Coughing",
  "Footsteps",
  "Laughing",
  "Brushing teeth",
  "Snoring",
  "Drinking sipping",

  // Interior / domestic (30–39)
  "Door knock",
  "Mouse click",
  "Keyboard typing",
  "Door wood creaks",
  "Can opening",
  "Washing machine",
  "Vacuum cleaner",
  "Clock alarm",
  "Clock tick",
  "Glass breaking",

  // Exterior / urban (40–49)
  "Helicopter",
  "Chainsaw",
  "Siren",
  "Car horn",
  "Engine",
  "Train",
  "Church bells",
  "Airplane",
  "Fireworks",
  "Hand saw",
] as const;

/** Type representing a valid ESC-50 label string */
export type ESC50Label = (typeof ESC50_LABELS)[number];

/** Number of classes in the ESC-50 dataset */
export const NUM_CLASSES = ESC50_LABELS.length; // 50

/**
 * Look up a class label by its integer index.
 * Returns "Unknown" if the index is out of range.
 */
export function getLabelByIndex(index: number): string {
  return ESC50_LABELS[index] ?? "Unknown";
}

/**
 * Category groupings for display purposes.
 * Maps a category name to the range of class indices it covers.
 */
export const LABEL_CATEGORIES: Record<string, { start: number; end: number; color: string }> = {
  Animals: { start: 0, end: 9, color: "#4ade80" },
  "Nature & Water": { start: 10, end: 19, color: "#38bdf8" },
  "Human Sounds": { start: 20, end: 29, color: "#f472b6" },
  "Indoor / Domestic": { start: 30, end: 39, color: "#fb923c" },
  "Urban / Outdoor": { start: 40, end: 49, color: "#a78bfa" },
};

/**
 * Returns the category name for a given class index.
 */
export function getCategoryForIndex(index: number): string {
  for (const [category, range] of Object.entries(LABEL_CATEGORIES)) {
    if (index >= range.start && index <= range.end) return category;
  }
  return "Other";
}

/**
 * Returns the display color for a given class index.
 */
export function getColorForIndex(index: number): string {
  for (const range of Object.values(LABEL_CATEGORIES)) {
    if (index >= range.start && index <= range.end) return range.color;
  }
  return "#94a3b8";
}
