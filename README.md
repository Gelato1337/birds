# Fieldmark 🐦

Point your phone at a Finnish bird or mammal — or record its call — and it tells you
what it is, right there in the browser, no server round-trip. Log it, rack up points,
chase missions, and compete with whoever you've roped into spotting with you. 313
species, birds and mammals.

PWA on GitHub Pages; Supabase handles the shared stuff (login, cross-device sync,
leaderboard). You can also **continue without logging in** — local-only mode keeps
every sighting in the device's cache and never touches a server.

## Run your own

The app is a static PWA — serve `index.html` from anywhere (GitHub Pages works).
For login + sync + leaderboard, spin up a free Supabase project and run the three
SQL scripts in [`db/`](db/); full walkthrough in [`db/SETUP.md`](db/SETUP.md). Skip
all of it and use local-only mode if you just want to try it.

## Screenshots

<table>
  <tr>
    <td align="center"><img src="docs/screenshots/spot.png" width="220" alt="Spot — photo species ID"><br><sub><b>Spot</b> — photo → species ID</sub></td>
    <td align="center"><img src="docs/screenshots/sound.png" width="220" alt="Sound ID"><br><sub><b>Sound</b> — record a call → BirdNET</sub></td>
    <td align="center"><img src="docs/screenshots/journal.png" width="220" alt="Journal"><br><sub><b>Journal</b> — your recorded species</sub></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/screenshots/quests.png" width="220" alt="Quests"><br><sub><b>Quests</b> — missions & assignments</sub></td>
    <td align="center"><img src="docs/screenshots/map.png" width="220" alt="Map"><br><sub><b>Map</b> — where you've spotted</sub></td>
    <td align="center"><img src="docs/screenshots/species-page.png" width="220" alt="Species page"><br><sub><b>Species page</b> — photo, call, stats, map</sub></td>
  </tr>
</table>

## What you can do

- 📸 Snap a photo → species ID
- 🔊 Record a call, or drop in an audio/video clip → species ID via BirdNET
- 📓 Build a dex, see where things were spotted and who found what
- 🎯 Missions: first bird, ID-by-sound, co-op spots, and ones that stay locked until
  you've earned them
- 🗺️ Map, leaderboard, works offline

## The dataset

Most "is this a bird?" apps lean on a plain CNN classifier, and that bugged me: a
classifier tells you *there's a bird* but not *where it is*, and it falls apart the
moment two animals share a frame. So this is built for **YOLO** — every training image
carries **bounding boxes around the actual animals**, so the model learns to point at
each one, not just shrug "yep, bird."

Building it, per species:

1. **Pull ~1000 photos** from open sources — iNaturalist first (community-verified IDs),
   then GBIF (live observations only, no pinned museum specimens), then Wikimedia for
   the rare stragglers.
2. **Auto-box every photo** with a general detector (YOLO11x): a tight box on each
   animal. No animal found → **tossed**. Lazy whole-frame "boxes" → tossed too.
3. **Label each box** with the species the photo was pulled for.
4. **The only thing we hand-label**: photos that show up in *more than one* species'
   pile — the exact same image pulled for two species. That's either a mis-ID or a frame
   with two different animals in it, so a human looks at it and says which **box belongs
   to which animal**. Everything else is automatic.
5. **Top up** the thin species until everyone has enough.

Out comes a clean, species-labeled detection set: hundreds of real, boxed examples per
species.

```mermaid
flowchart TD
  A["iNaturalist / GBIF / Wikimedia<br/>~1000 photos per species"] --> B[YOLO11x auto-boxes every animal]
  B -->|no animal / whole-frame box| X[Tossed]
  B -->|clean box + species| D[Species-labeled dataset]
  B -->|same photo in 2 species' piles| M["Hand-label:<br/>which box is which animal"]
  M --> D
  D --> E["Augment<br/>flip · colour · scale · mosaic"]
  E --> F["Train fieldmark-s (YOLO)<br/>313 species, GPU"]
  F --> G[Export to ONNX → runs in the browser]
```

## Models

Both run on-device in the browser (WebGPU, falling back to WASM):

- **fieldmark-s** — *our* YOLO detector, trained on the dataset above. This is the one
  we build and retrain.
- **BirdNET** — an off-the-shelf, already-pretrained bird-sound model; we just map its
  output onto our 313 species.

Hand-label more shared photos in the app → they feed back into the dataset → retrain
fieldmark-s.
