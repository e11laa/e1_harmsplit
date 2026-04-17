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

    StftOlaProcessor();

    void prepare(int maxBlockSize);
    void reset();

    void processBlock(const float* input, float* output, int numSamples);

private:
    void processAvailableFrames();
    void flushReadySamples();
    float measureIfftScale();

    juce::dsp::FFT fft;
    juce::dsp::WindowingFunction<float> window;

    std::array<float, fftSize> windowTable{};
    std::array<float, fftSize> windowSquared{};
    std::array<float, fftSize * 2> fftData{};

    std::vector<float> inputQueue;
    std::vector<float> olaQueue;
    std::vector<float> normQueue;
    std::vector<float> outputQueue;

    int nextFrameStart = 0;
    float ifftScale = 1.0f;
};
