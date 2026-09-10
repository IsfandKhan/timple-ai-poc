# Character Spec — "Mara Vance"

Fixed identity. Every generation references this. Do not drift the description
between scenes — only pose, wardrobe layer, location, light, and expression change.

## Trigger word

`mara_vance` (LoRA). Use as: `a photo of mara_vance woman, ...`

## Canonical description (use verbatim in prompts)

The scripts read the `>`-quoted lines below and prepend the trigger word (Flux) or
use them as-is (SDXL). Do not lead this block with the trigger word.

> a 34-year-old woman of Greek heritage, warm olive skin, oval face with a
> defined jaw, deep-set hazel eyes, thick dark eyebrows with a small vertical scar
> through the right brow, straight nose, full lips, faint freckles across the nose,
> dark brown wavy hair to the shoulders usually tied back in a low bun with loose
> strands, lean athletic build, 5 feet 7

## Consistent details

| Trait | Value |
|---|---|
| Age read | early-to-mid 30s |
| Hair | dark brown, wavy, shoulder length, low bun + loose strands (down only in 2 scenes) |
| Eyes | hazel, deep-set |
| Distinguishing mark | small vertical scar through right eyebrow |
| Skin | warm olive, faint nose freckles, minimal makeup |
| Build | lean, athletic, 5'7" |
| Signature item | small tarnished silver coin pendant on a leather cord (present every scene) |
| Base wardrobe | faded olive field jacket, white cotton tee, dark canvas trousers, scuffed brown leather boots |

## Wardrobe layers (only variation allowed)

- Base (jacket on)
- Jacket off / sleeves rolled (heat, indoor)
- Rain shell over jacket (storm scene)
- Wool sweater under jacket (night, cold)

## Persona (for expression direction)

Field archaeologist, documentary subject. Composed, observant, dry humor. Not a
model — real expressions: concentration, mild fatigue, quiet satisfaction, alertness.

## Hero image

- File: `character/hero.png`  (= `hero_candidates/hero_01.png`)
- Locked seed: `1137`
- Prompt: `a photo of <canonical description>, head and shoulders portrait, front view, neutral, soft window light, plain studio backdrop. photorealistic, 35mm portrait, natural light, realistic skin texture, film grain`
- Base model for hero: Flux.1-dev (fp8), no LoRA, no ControlNet

## Negative / avoid

`glamour, heavy makeup, airbrushed skin, perfect symmetry, influencer pose, duck lips,
plastic skin, over-sharpened, teenager, elderly, blonde, blue eyes, cluttered face tattoos`
