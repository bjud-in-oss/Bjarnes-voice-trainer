# Architecture Overview

## Core Technology
- **Frontend Framework:** React 18 (TypeScript).
- **Styling:** Tailwind CSS (via CDN for portability).
- **ML Engine:** TensorFlow.js (`@tensorflow/tfjs`).
- **Audio Model:** `@tensorflow-models/speech-commands`.

## Data Flow
1.  **Input:** Microphone audio stream via Web Audio API.
2.  **Preprocessing:** `speech-commands` library converts raw audio into spectrograms (FFT).
3.  **Transfer Learning:** A lightweight dense neural network is trained on top of the frozen pre-trained features.
4.  **Inference:** Real-time classification loop outputs probability distribution across 4 classes.
5.  **Feedback:** React state updates UI based on probability thresholds (>85%).

## State Management
- **Local State:** React `useState` and `useRef` manage the TFJS model instances and training loop.
- **Persistence:** Models are serialized to JSON/Binary and saved via browser download or LocalStorage.
