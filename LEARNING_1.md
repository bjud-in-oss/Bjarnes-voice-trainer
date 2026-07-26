# Learning Log 01: Audio Classification for Aphasia

## Context
Standard LLMs (like Whisper) transcribe speech to text. For Bjarne, who uses non-verbal sounds ("Hm-hm"), these models often filter out the sounds as "noise" or "fillers". We need a raw acoustic classifier, not a linguistic one.

## Technical Decisions
1.  **Browser-based Training:** We use `tfjs` to train locally. This ensures:
    - **Privacy:** Audio never leaves the device.
    - **Latency:** Immediate feedback (green light) is crucial for user confidence.
    - **Customization:** The "Hm-hm" sound is unique to the individual.

2.  **Transfer Learning:** Instead of training a CNN from scratch, we use the `BROWSER_FFT` model. It already knows how to "hear" frequencies; we just teach it to map those frequencies to our specific labels.

3.  **UX Challenges:** The training process must be understandable without technical jargon. "Hold to Record" is more intuitive than "Start/Stop" for capturing short bursts of sound.
