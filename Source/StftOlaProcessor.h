#pragma once

#include <JuceHeader.h>
#include <juce_dsp/juce_dsp.h>

#include <array>
#include <vector>

class StftOlaProcessor final
{
public:
    static constexpr int fftOrder = 11;
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr int overlapFactor = 4;
    static constexpr int hopSize = fftSize / overlapFactor;

    // With center-less streaming OLA, one frame minus one hop is buffered before output is available.
    static constexpr int reportedLatencySamples = fftSize - hopSize;
    static constexpr float minTrackedPitchHz = 40.0f;
    static constexpr float maxTrackedPitchHz = 2000.0f;

    StftOlaProcessor();

    void prepare(double sampleRate, int maxBlockSize);
    void reset();
    void setDebugOutput(bool shouldOutput, juce::String label);

    void processBlock(const float* input, float* output, int numSamples);
    float getSmoothedF0Hz() const noexcept;

private:
    static constexpr float silencePowerThreshold = 1.0e-7f;
    static constexpr float spectralContrastThreshold = 6.0f;
    static constexpr int medianFilterLength = 5;
    static constexpr int debugPrintIntervalFrames = 24;

    void processAvailableFrames();
    void flushReadySamples();
    void analysePitchFrame(float framePower);
    void updatePitchSmoothing(float candidateHz);
    float applyMedianFilter(float candidateHz);
    void maybePrintDebugPitch();
    float measureIfftScale();

    juce::dsp::FFT fft;
    juce::dsp::WindowingFunction<float> window;

    std::array<float, fftSize> windowTable{};
    std::array<float, fftSize> windowSquared{};
    std::array<float, fftSize * 2> fftData{};
    std::array<float, (fftSize / 2) + 1> magnitudeSpectrum{};
    std::array<float, medianFilterLength> medianHistory{};

    std::vector<float> inputQueue;
    std::vector<float> olaQueue;
    std::vector<float> normQueue;
    std::vector<float> outputQueue;

    double sampleRateHz = 48000.0;
    int nextFrameStart = 0;
    int minTrackedBin = 1;
    int maxTrackedBin = (fftSize / 2) - 1;
    int medianWriteIndex = 0;
    int medianHistoryCount = 0;
    int debugFrameCounter = 0;
    float ifftScale = 1.0f;
    float smoothedF0Hz = 0.0f;
    float smoothingAlpha = 0.85f;
    bool debugOutputEnabled = false;
    juce::String debugLabel = "F0";
};
