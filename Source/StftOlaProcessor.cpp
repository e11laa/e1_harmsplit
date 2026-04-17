#include "StftOlaProcessor.h"

#include <algorithm>
#include <cmath>

StftOlaProcessor::StftOlaProcessor()
    : fft(fftOrder),
      window(fftSize, juce::dsp::WindowingFunction<float>::hann, false)
{
    windowTable.fill(1.0f);
    window.multiplyWithWindowingTable(windowTable.data(), fftSize);

    for (int sample = 0; sample < fftSize; ++sample)
        windowSquared[sample] = windowTable[sample] * windowTable[sample];

    ifftScale = measureIfftScale();
}

void StftOlaProcessor::prepare(int maxBlockSize)
{
    reset();

    const auto reserveSize = juce::jmax(fftSize * 2, maxBlockSize + fftSize + hopSize);
    inputQueue.reserve(static_cast<size_t>(reserveSize));
    olaQueue.reserve(static_cast<size_t>(reserveSize));
    normQueue.reserve(static_cast<size_t>(reserveSize));
    outputQueue.reserve(static_cast<size_t>(reserveSize));
}

void StftOlaProcessor::reset()
{
    inputQueue.clear();
    olaQueue.clear();
    normQueue.clear();
    outputQueue.clear();
    nextFrameStart = 0;
}

void StftOlaProcessor::processBlock(const float* input, float* output, int numSamples)
{
    if (numSamples <= 0)
        return;

    inputQueue.insert(inputQueue.end(), input, input + numSamples);

    processAvailableFrames();
    flushReadySamples();

    if (static_cast<int>(outputQueue.size()) < numSamples)
    {
        const auto missingSamples = numSamples - static_cast<int>(outputQueue.size());
        outputQueue.insert(outputQueue.end(), static_cast<size_t>(missingSamples), 0.0f);
    }

    std::copy_n(outputQueue.begin(), numSamples, output);
    outputQueue.erase(outputQueue.begin(), outputQueue.begin() + numSamples);
}

void StftOlaProcessor::processAvailableFrames()
{
    while (static_cast<int>(inputQueue.size()) - nextFrameStart >= fftSize)
    {
        std::fill(fftData.begin(), fftData.end(), 0.0f);

        for (int sample = 0; sample < fftSize; ++sample)
            fftData[sample] = inputQueue[static_cast<size_t>(nextFrameStart + sample)] * windowTable[sample];

        fft.performRealOnlyForwardTransform(fftData.data());
        fft.performRealOnlyInverseTransform(fftData.data());

        const auto requiredSize = static_cast<size_t>(nextFrameStart + fftSize);

        if (olaQueue.size() < requiredSize)
        {
            olaQueue.resize(requiredSize, 0.0f);
            normQueue.resize(requiredSize, 0.0f);
        }

        for (int sample = 0; sample < fftSize; ++sample)
        {
            const auto writeIndex = static_cast<size_t>(nextFrameStart + sample);

            olaQueue[writeIndex] += fftData[sample] * windowTable[sample];
            normQueue[writeIndex] += windowSquared[sample] * ifftScale;
        }

        nextFrameStart += hopSize;
    }
}

void StftOlaProcessor::flushReadySamples()
{
    const auto numReady = nextFrameStart;

    if (numReady <= 0)
        return;

    outputQueue.reserve(outputQueue.size() + static_cast<size_t>(numReady));

    for (int sample = 0; sample < numReady; ++sample)
    {
        const auto denominator = normQueue[static_cast<size_t>(sample)];
        const auto reconstructed = denominator > 1.0e-9f ? olaQueue[static_cast<size_t>(sample)] / denominator : 0.0f;
        outputQueue.push_back(reconstructed);
    }

    inputQueue.erase(inputQueue.begin(), inputQueue.begin() + numReady);
    olaQueue.erase(olaQueue.begin(), olaQueue.begin() + numReady);
    normQueue.erase(normQueue.begin(), normQueue.begin() + numReady);
    nextFrameStart -= numReady;
}

float StftOlaProcessor::measureIfftScale()
{
    std::fill(fftData.begin(), fftData.end(), 0.0f);

    fftData[0] = 1.0f;
    fft.performRealOnlyForwardTransform(fftData.data());
    fft.performRealOnlyInverseTransform(fftData.data());

    const auto measuredScale = std::abs(fftData[0]);
    return measuredScale > 1.0e-12f ? measuredScale : 1.0f;
}
