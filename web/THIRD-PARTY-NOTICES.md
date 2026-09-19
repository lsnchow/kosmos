# Third-party notices

## Mirage (design language and landing-page patterns)

Nightshift's console theme and landing page adapt the visual language of **Mirage**
(https://github.com/kyan-yang/Mirage), a TreeHacks 2026 project by Shrey
Birmiwal, Kyan Yang, Kevin Thomas and Adi Prasad.

What was adapted: the dark palette and grayscale ramp, the single-accent
treatment, the display/body/mono type pairing, the fullscreen-video landing
treatment with a dark overlay, the centered tab navigation with an accent
underline, and the example-gallery card pattern.

Three component *shapes* were also ported, rewritten rather than copied:

| Nightshift | Adapted from |
|---|---|
| `TabSelector` | `TabSelector.tsx` — generalised, with `role="tablist"`, `aria-selected` and arrow-key navigation added |
| `StageLadder` | `ProgressSteps.tsx` — the labelled left-rail pipeline, re-aimed at Nightshift's four-stage Chain and given a "not reported" state |
| `GalleryModal` | `GalleryModal.tsx` — keeps its portal-to-`document.body` fix (commit `db98e0f`), adds a focus trap and focus restoration |

What was **not** taken: none of Mirage's stylesheet, pipeline, models, or data,
and none of its application logic. Nightshift's state, data model, copy and
accessibility work are its own.

Mirage is MIT licensed. Its notice is reproduced in full below as that licence
requires.

```
MIT License

Copyright (c) 2026 Kyan Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Fonts

Self-hosted under `web/public/fonts/` rather than loaded from a CDN, so the
console renders identically without network access — a stage requirement, not a
preference.

| Family | Files | Licence |
|---|---|---|
| Geist Sans | `Geist-Regular/Medium/SemiBold.woff2` | SIL Open Font License 1.1, © 2023 Vercel |
| Geist Mono | `GeistMono-Regular/Medium.woff2` | SIL Open Font License 1.1, © 2023 Vercel |
| Bitcount Grid Double | `BitcountGridDouble-vf.woff2` | SIL Open Font License 1.1 |

The SIL OFL permits redistribution of the font files with this notice. The fonts
are used as-is and are not modified or renamed.
