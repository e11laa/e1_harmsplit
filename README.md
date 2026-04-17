# Harmonic Split (JUCE)

Harmonic Split is a JUCE-based spectral audio effect inspired by Bitwig Studio's Harmonic Split concept.
The plugin performs real-time STFT analysis, estimates fundamental frequency (F0), separates spectral content into three layers, and outputs the selected layer through a single stereo output.

## Current Design (Single Output, Layered Mode)

This repository currently uses a single stereo output design for better DAW usability:

- Input: 1 stereo bus
- Output: 1 stereo bus
- Layer selection via Output Mode parameter:
  - All
  - Harmonics A (fundamental + odd harmonics)
  - Harmonics B (even harmonics)
  - Non-harmonics

This replaces the previous multi-output routing approach.

## Feature Summary

- STFT/IFFT pipeline with overlap-add reconstruction
- Real-time F0 tracking in frequency domain
- Harmonic masking with tolerance width control
- Harmonics A/B balance control
- Per-layer gain controls
- APVTS-based parameter/state management
- Simple editor UI with rotary controls and Output Mode selector

## Parameters

| ID | Name | Type | Range | Default | Notes |
|---|---|---|---|---|---|
| smoothness | Smoothness | Float | 0.0 to 100.0 ms | 10.0 ms | Pitch smoothing response |
| harmonic_width | Harmonic Width | Float | 0.1 to 2.0 | 1.0 | Mask tolerance width scale |
| harmonics_balance | Harmonics Balance | Float | -1.0 to 1.0 | 0.0 | Redistributes odd/even energy between A/B |
| gain_a | Gain A | Float | -96.0 to +12.0 dB | 0.0 dB | Harmonics A output gain |
| gain_b | Gain B | Float | -96.0 to +12.0 dB | 0.0 dB | Harmonics B output gain |
| gain_nonharm | Gain Nonharm | Float | -96.0 to +12.0 dB | 0.0 dB | Non-harmonics output gain |
| outputMode | Output Mode | Choice | All / Harmonics A / Harmonics B / Non-harmonics | All | Selects final layer sent to output |

## Build

### Requirements

- CMake 3.22+
- C++20 compiler
- JUCE 8 (fetched automatically by default)
- On Windows: Visual Studio 2022 or newer with C++ workload

### Configure

```powershell
cmake -S . -B build
```

### Build shared code target

```powershell
cmake --build build --config Release --target HarmonicSplit
```

### Build VST3 target

```powershell
cmake --build build --config Release --target HarmonicSplit_VST3
```

## JUCE Source Selection

By default, JUCE is fetched using FetchContent.
You can also point to a local JUCE checkout:

```powershell
cmake -S . -B build -DHARMONIC_SPLIT_USE_LOCAL_JUCE=ON -DJUCE_DIR="C:/path/to/JUCE"
```

## Runtime Notes

- The plugin reports latency from STFT overlap processing.
- Pitch logging currently appends text to Desktop/PitchLog.txt at sample-throttled intervals.
- Output Mode controls which separated layer is heard from the single stereo output.

## Troubleshooting

### Permission denied while building HarmonicSplit_VST3 on Windows

If build fails at the final install/copy step to Program Files/Common Files/VST3, run your terminal/IDE with Administrator privileges, or build the shared-code target first:

```powershell
cmake --build build --config Release --target HarmonicSplit
```

This confirms compile/link success even if system-level copy permissions are restricted.

## Repository Layout

- CMakeLists.txt: project and JUCE build setup
- Source/PluginProcessor.h, Source/PluginProcessor.cpp: DSP, APVTS, processing logic
- Source/PluginEditor.h, Source/PluginEditor.cpp: editor UI and attachments

## Development Status

Core implementation through Step 5 is in place:

- Step 1: Project setup and routing
- Step 2: STFT/IFFT + reconstruction
- Step 3: F0 tracking
- Step 4: Harmonic spectral masking and layer separation
- Step 5: APVTS and GUI controls

Further improvements can include:

- Asynchronous/non-audio-thread pitch logging
- Parameter automation polish and UI metering
- Additional mask shapes and transient handling
- Performance tuning and profiling across DAWs
