# Mianxiang — Traditional Chinese Face Reading

A web app that reads an uploaded portrait using traditional Chinese physiognomy (面相 / mianxiang). All analysis runs locally in the browser — the photo never leaves the device.

## How it works

1. **Landmark detection** — MediaPipe Face Landmarker (loaded from CDN, runs in-browser via WASM) finds 478 facial landmarks in the uploaded photo.
2. **Geometry** — [app.js](app.js) computes classical mianxiang measurements from those landmarks: the Three Courts (三停) vertical proportions, face aspect and jaw ratios, eye size/spacing, brow thickness and angle, nose length/width, mouth width, lip fullness, philtrum, chin, and left/right asymmetries (mouth tilt, eye openness, brow height — measured after de-rotating for head tilt). The mesh ends mid-forehead, so the hairline is extrapolated upward for the upper court. Each measurement is converted to a z-score against population statistics (the `STATS` table, calibrated from real phone photographs — important, because phone lenses systematically distort geometry relative to paintings or studio portraits), so readings reflect how *this* face differs from a typical photo rather than absolute pixel ratios.
3. **Interpretation** — the measurements are mapped to traditional readings:
   - **Five-element face type** (五行): Wood, Fire, Earth, Metal, or Water, from overall face shape
   - **Three Courts** (三停): youth / middle years / later years
   - **Five Features** (五官): brows, eyes, nose, mouth, plus forehead and chin
   - **Balance & Symmetry** (阴阳): mouth tilt, uneven eyes (阴阳眼), brow height, with a combined verdict
   - **Palaces of the face** (十二宫): a selection of eight palaces keyed to measurable regions
   - **Other Mirrors** (他山之镜): the same measurements read through Vedic Samudrika Shastra, the Greek four temperaments, and Japanese ninsō (人相)
4. An annotated canvas overlays the court lines and feature points on the photo.

Readings are deterministic per photo but vary between faces: band thresholds are z-score based, texts have multiple phrasing variants chosen by a PRNG seeded from the face's own geometry, and the summary names the two statistically most distinctive features of that particular face.

## Sharing

- **Share result** composes a 1080×1500 share-card PNG (cropped annotated photo, element type, courts chart, trait chips, disclaimer) and opens the native share sheet where available, otherwise downloads the image.
- **Copy reading** puts the full reading on the clipboard as plain text.

## Running

Any static file server works, e.g.:

```bash
npx -y http-server -p 8742 .
```

Then open http://localhost:8742. (`.claude/launch.json` is configured for the Claude Code browser preview.) An internet connection is needed on first load to fetch the MediaPipe model (~3 MB); the photo analysis itself is fully local.

## Files

- `index.html` — page structure
- `styles.css` — ink-and-cinnabar dark theme
- `app.js` — landmark detection, measurement engine, interpretation content, rendering

## Disclaimer

Mianxiang is a folk tradition offered here as cultural exploration and entertainment. It has no scientific basis — facial features do not determine character, fortune, or destiny.
