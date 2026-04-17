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

    magnitudeSpectrum.fill(0.0f);
    medianHistory.fill(0.0f);

    ifftScale = measureIfftScale();
}

void StftOlaProcessor::prepare(double sampleRate, int maxBlockSize)
{
    sampleRateHz = sampleRate > 0.0 ? sampleRate : 48000.0;

    constexpr double smoothingTauSeconds = 0.08;
    const auto framePeriodSeconds = static_cast<double>(hopSize) / sampleRateHz;

    smoothingAlpha = static_cast<float>(std::exp(-framePeriodSeconds / smoothingTauSeconds));
    smoothingAlpha = juce::jlimit(0.0f, 0.9995f, smoothingAlpha);

    const auto maxSearchBin = (fftSize / 2) - 1;

    minTrackedBin = juce::jlimit(
        1,
        maxSearchBin - 2,
        static_cast<int>(std::floor((minTrackedPitchHz * static_cast<float>(fftSize)) / static_cast<float>(sampleRateHz))));

    maxTrackedBin = juce::jlimit(
        minTrackedBin + 2,
        maxSearchBin,
        static_cast<int>(std::ceil((maxTrackedPitchHz * static_cast<float>(fftSize)) / static_cast<float>(sampleRateHz))));

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

    medianHistory.fill(0.0f);

    nextFrameStart = 0;
    medianWriteIndex = 0;
    medianHistoryCount = 0;
    debugFrameCounter = 0;
    smoothedF0Hz = 0.0f;
}

void StftOlaProcessor::setDebugOutput(bool shouldOutput, juce::String label)
{
    debugOutputEnabled = shouldOutput;

    if (label.isNotEmpty())
        debugLabel = label;

    debugFrameCounter = 0;
}

float StftOlaProcessor::getSmoothedF0Hz() const noexcept
{
    return smoothedF0Hz;
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

        float framePower = 0.0f;

        for (int sample = 0; sample < fftSize; ++sample)
        {
            const auto inputSample = inputQueue[static_cast<size_t>(nextFrameStart + sample)];

            fftData[sample] = inputSample * windowTable[sample];
            framePower += inputSample * inputSample;
        }

        framePower *= (1.0f / static_cast<float>(fftSize));

        fft.performRealOnlyForwardTransform(fftData.data());
        analysePitchFrame(framePower);
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

void StftOlaProcessor::analysePitchFrame(float framePower)
{
    float candidateHz = 0.0f;

    if (framePower > silencePowerThreshold)
    {
        constexpr auto nyquistBin = fftSize / 2;

        float bandEnergy = 0.0f;

        magnitudeSpectrum[0] = fftData[0] * fftData[0];
        magnitudeSpectrum[nyquistBin] = fftData[1] * fftData[1];

        for (int bin = 1; bin < nyquistBin; ++bin)
        {
            const auto real = fftData[2 * bin];
            const auto imaginary = fftData[2 * bin + 1];
            const auto magnitude = (real * real) + (imaginary * imaginary);

            magnitudeSpectrum[static_cast<size_t>(bin)] = magnitude;

            if (bin >= minTrackedBin && bin <= maxTrackedBin)
                bandEnergy += magnitude;
        }

        float bestScore = 0.0f;
        int bestBin = -1;

        for (int bin = minTrackedBin + 1; bin < maxTrackedBin; ++bin)
        {
            const auto left = magnitudeSpectrum[static_cast<size_t>(bin - 1)];
            const auto center = magnitudeSpectrum[static_cast<size_t>(bin)];
            const auto right = magnitudeSpectrum[static_cast<size_t>(bin + 1)];

            if (center <= left || center < right)
                continue;

            auto score = center;

            if ((2 * bin) <= nyquistBin)
                score += 0.5f * magnitudeSpectrum[static_cast<size_t>(2 * bin)];

            if ((3 * bin) <= nyquistBin)
                score += 0.25f * magnitudeSpectrum[static_cast<size_t>(3 * bin)];

            if (score > bestScore)
            {
                bestScore = score;
                bestBin = bin;
            }
        }

        const auto trackedBinCount = juce::jmax(1, maxTrackedBin - minTrackedBin + 1);
        const auto averageBandEnergy = bandEnergy / static_cast<float>(trackedBinCount);

        if (bestBin > 0 && bestScore > (averageBandEnergy * spectralContrastThreshold))
        {
            auto refinedBin = static_cast<float>(bestBin);

            const auto y1 = magnitudeSpectrum[static_cast<size_t>(bestBin - 1)];
            const auto y2 = magnitudeSpectrum[static_cast<size_t>(bestBin)];
            const auto y3 = magnitudeSpectrum[static_cast<size_t>(bestBin + 1)];

            const auto denominator = y1 - (2.0f * y2) + y3;

            if (std::abs(denominator) > 1.0e-12f)
            {
                const auto delta = 0.5f * (y1 - y3) / denominator;
                refinedBin += juce::jlimit(-0.5f, 0.5f, delta);
            }

            candidateHz = refinedBin * static_cast<float>(sampleRateHz / static_cast<double>(fftSize));

            const auto halfBin = bestBin / 2;

            if (halfBin >= minTrackedBin)
            {
                const auto halfMagnitude = magnitudeSpectrum[static_cast<size_t>(halfBin)];
                const auto fundamentalMagnitude = magnitudeSpectrum[static_cast<size_t>(bestBin)];

                if (halfMagnitude > (fundamentalMagnitude * 0.72f))
                    candidateHz *= 0.5f;
            }

            if (candidateHz < minTrackedPitchHz || candidateHz > maxTrackedPitchHz)
                candidateHz = 0.0f;
        }
    }

    updatePitchSmoothing(candidateHz);
    maybePrintDebugPitch();
}

void StftOlaProcessor::updatePitchSmoothing(float candidateHz)
{
    if (candidateHz <= 0.0f)
    {
        medianWriteIndex = 0;
        medianHistoryCount = 0;

        smoothedF0Hz *= 0.9f;

        if (smoothedF0Hz < (minTrackedPitchHz * 0.25f))
            smoothedF0Hz = 0.0f;

        return;
    }

    if (smoothedF0Hz > 0.0f)
    {
        while (candidateHz > (smoothedF0Hz * 1.9f))
            candidateHz *= 0.5f;

        while (candidateHz < (smoothedF0Hz / 1.9f) && (candidateHz * 2.0f) <= maxTrackedPitchHz)
            candidateHz *= 2.0f;
    }

    const auto medianHz = applyMedianFilter(candidateHz);

    if (smoothedF0Hz <= 0.0f)
    {
        smoothedF0Hz = medianHz;
        return;
    }

    smoothedF0Hz = (smoothingAlpha * smoothedF0Hz) + ((1.0f - smoothingAlpha) * medianHz);
}

float StftOlaProcessor::applyMedianFilter(float candidateHz)
{
    medianHistory[static_cast<size_t>(medianWriteIndex)] = candidateHz;
    medianWriteIndex = (medianWriteIndex + 1) % medianFilterLength;
    medianHistoryCount = juce::jmin(medianHistoryCount + 1, medianFilterLength);

    std::array<float, medianFilterLength> sorted{};
    std::copy_n(medianHistory.begin(), static_cast<size_t>(medianHistoryCount), sorted.begin());
    std::sort(sorted.begin(), sorted.begin() + medianHistoryCount);

    return sorted[static_cast<size_t>(medianHistoryCount / 2)];
}

void StftOlaProcessor::maybePrintDebugPitch()
{
    if (!debugOutputEnabled)
        return;

    ++debugFrameCounter;

    if (debugFrameCounter < debugPrintIntervalFrames)
        return;

    debugFrameCounter = 0;

    if (smoothedF0Hz > 0.0f)
        DBG(debugLabel << ": " << juce::String(smoothedF0Hz, 2) << " Hz");
    else
        DBG(debugLabel << ": --");
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
