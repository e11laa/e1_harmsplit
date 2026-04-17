#include "PluginProcessor.h"
#include "PluginEditor.h"

#include <algorithm>
#include <cmath>

namespace
{
float toSmoothingAlpha(double sampleRateHz, int hopSize, float smoothnessMs)
{
    if (smoothnessMs <= 0.0f || sampleRateHz <= 0.0)
        return 0.0f;

    const auto tauSeconds = static_cast<double>(smoothnessMs) * 0.001;
    const auto framePeriodSeconds = static_cast<double>(hopSize) / sampleRateHz;
    const auto alpha = static_cast<float>(std::exp(-framePeriodSeconds / std::max(1.0e-9, tauSeconds)));

    return juce::jlimit(0.0f, 0.9995f, alpha);
}
}

ChannelSpectralSplitter::ChannelSpectralSplitter()
    : fft(fftOrder),
      window(fftSize, juce::dsp::WindowingFunction<float>::hann, false)
{
    analysisWindow.fill(1.0f);
    window.multiplyWithWindowingTable(analysisWindow.data(), fftSize);

    for (int sample = 0; sample < fftSize; ++sample)
        windowSquared[sample] = analysisWindow[sample] * analysisWindow[sample];

    magnitudeSpectrum.fill(0.0f);
    maskA.fill(0.0f);
    maskB.fill(0.0f);
    maskN.fill(1.0f);
    medianHistory.fill(0.0f);

    ifftScale = measureIfftScale();
}

void ChannelSpectralSplitter::prepare(double sampleRate, int maxBlockSize)
{
    sampleRateHz = sampleRate > 0.0 ? sampleRate : 48000.0;
    binWidthHz = static_cast<float>(sampleRateHz / static_cast<double>(fftSize));

    const auto nyquistBin = (fftSize / 2) - 1;

    minTrackedBin = juce::jlimit(
        1,
        nyquistBin - 2,
        static_cast<int>(std::floor(minTrackedPitchHz / binWidthHz)));

    maxTrackedBin = juce::jlimit(
        minTrackedBin + 2,
        nyquistBin,
        static_cast<int>(std::ceil(maxTrackedPitchHz / binWidthHz)));

    setRuntimeParameters(smoothnessMs, harmonicWidth, harmonicsBalance);
    reset();

    const auto reserveSize = juce::jmax(fftSize * 2, maxBlockSize + fftSize + hopSize);

    inputQueue.reserve(static_cast<size_t>(reserveSize));
    olaQueueA.reserve(static_cast<size_t>(reserveSize));
    olaQueueB.reserve(static_cast<size_t>(reserveSize));
    olaQueueN.reserve(static_cast<size_t>(reserveSize));
    normQueue.reserve(static_cast<size_t>(reserveSize));
    outputQueueA.reserve(static_cast<size_t>(reserveSize));
    outputQueueB.reserve(static_cast<size_t>(reserveSize));
    outputQueueN.reserve(static_cast<size_t>(reserveSize));
}

void ChannelSpectralSplitter::reset()
{
    inputQueue.clear();
    olaQueueA.clear();
    olaQueueB.clear();
    olaQueueN.clear();
    normQueue.clear();
    outputQueueA.clear();
    outputQueueB.clear();
    outputQueueN.clear();

    medianHistory.fill(0.0f);
    maskA.fill(0.0f);
    maskB.fill(0.0f);
    maskN.fill(1.0f);

    nextFrameStart = 0;
    medianWriteIndex = 0;
    medianHistoryCount = 0;
    debugFrameCounter = 0;
    smoothedF0Hz = 0.0f;
}

void ChannelSpectralSplitter::processBlock(const float* input,
                                           float* harmonicsAOutput,
                                           float* harmonicsBOutput,
                                           float* nonHarmonicsOutput,
                                           int numSamples)
{
    if (numSamples <= 0)
        return;

    inputQueue.insert(inputQueue.end(), input, input + numSamples);

    processAvailableFrames();
    flushReadySamples();

    if (static_cast<int>(outputQueueA.size()) < numSamples)
    {
        const auto missingSamples = numSamples - static_cast<int>(outputQueueA.size());

        outputQueueA.insert(outputQueueA.end(), static_cast<size_t>(missingSamples), 0.0f);
        outputQueueB.insert(outputQueueB.end(), static_cast<size_t>(missingSamples), 0.0f);
        outputQueueN.insert(outputQueueN.end(), static_cast<size_t>(missingSamples), 0.0f);
    }

    std::copy_n(outputQueueA.begin(), numSamples, harmonicsAOutput);
    std::copy_n(outputQueueB.begin(), numSamples, harmonicsBOutput);
    std::copy_n(outputQueueN.begin(), numSamples, nonHarmonicsOutput);

    outputQueueA.erase(outputQueueA.begin(), outputQueueA.begin() + numSamples);
    outputQueueB.erase(outputQueueB.begin(), outputQueueB.begin() + numSamples);
    outputQueueN.erase(outputQueueN.begin(), outputQueueN.begin() + numSamples);
}

void ChannelSpectralSplitter::setRuntimeParameters(float newSmoothnessMs, float newHarmonicWidth, float newHarmonicsBalance)
{
    smoothnessMs = juce::jlimit(0.0f, 100.0f, newSmoothnessMs);
    harmonicWidth = juce::jlimit(0.1f, 2.0f, newHarmonicWidth);
    harmonicsBalance = juce::jlimit(-1.0f, 1.0f, newHarmonicsBalance);
    smoothingAlpha = toSmoothingAlpha(sampleRateHz, hopSize, smoothnessMs);
}

void ChannelSpectralSplitter::setDebugOutput(bool shouldOutput, juce::String label)
{
    debugOutputEnabled = shouldOutput;

    if (label.isNotEmpty())
        debugLabel = label;

    debugFrameCounter = 0;
}

float ChannelSpectralSplitter::getSmoothedF0Hz() const noexcept
{
    return smoothedF0Hz;
}

void ChannelSpectralSplitter::processAvailableFrames()
{
    while (static_cast<int>(inputQueue.size()) - nextFrameStart >= fftSize)
    {
        std::fill(inputSpectrum.begin(), inputSpectrum.end(), 0.0f);

        float framePower = 0.0f;

        for (int sample = 0; sample < fftSize; ++sample)
        {
            const auto inputSample = inputQueue[static_cast<size_t>(nextFrameStart + sample)];

            inputSpectrum[sample] = inputSample * analysisWindow[sample];
            framePower += inputSample * inputSample;
        }

        framePower *= (1.0f / static_cast<float>(fftSize));

        fft.performRealOnlyForwardTransform(inputSpectrum.data());

        const auto trackedF0Hz = estimateF0Hz(framePower);
        buildMasks(trackedF0Hz);
        applyMasksAndReconstructFrame();
        maybePrintDebugPitch();

        nextFrameStart += hopSize;
    }
}

void ChannelSpectralSplitter::flushReadySamples()
{
    const auto numReady = nextFrameStart;

    if (numReady <= 0)
        return;

    outputQueueA.reserve(outputQueueA.size() + static_cast<size_t>(numReady));
    outputQueueB.reserve(outputQueueB.size() + static_cast<size_t>(numReady));
    outputQueueN.reserve(outputQueueN.size() + static_cast<size_t>(numReady));

    for (int sample = 0; sample < numReady; ++sample)
    {
        const auto denominator = normQueue[static_cast<size_t>(sample)];

        if (denominator > 1.0e-9f)
        {
            outputQueueA.push_back(olaQueueA[static_cast<size_t>(sample)] / denominator);
            outputQueueB.push_back(olaQueueB[static_cast<size_t>(sample)] / denominator);
            outputQueueN.push_back(olaQueueN[static_cast<size_t>(sample)] / denominator);
        }
        else
        {
            outputQueueA.push_back(0.0f);
            outputQueueB.push_back(0.0f);
            outputQueueN.push_back(0.0f);
        }
    }

    inputQueue.erase(inputQueue.begin(), inputQueue.begin() + numReady);
    olaQueueA.erase(olaQueueA.begin(), olaQueueA.begin() + numReady);
    olaQueueB.erase(olaQueueB.begin(), olaQueueB.begin() + numReady);
    olaQueueN.erase(olaQueueN.begin(), olaQueueN.begin() + numReady);
    normQueue.erase(normQueue.begin(), normQueue.begin() + numReady);
    nextFrameStart -= numReady;
}

float ChannelSpectralSplitter::estimateF0Hz(float framePower)
{
    if (framePower <= silencePowerThreshold)
        return updateSmoothedF0(0.0f);

    constexpr int nyquistBin = fftSize / 2;

    float bandEnergy = 0.0f;

    magnitudeSpectrum[0] = std::abs(inputSpectrum[0]);
    magnitudeSpectrum[nyquistBin] = std::abs(inputSpectrum[1]);

    for (int bin = 1; bin < nyquistBin; ++bin)
    {
        const auto real = inputSpectrum[2 * bin];
        const auto imaginary = inputSpectrum[2 * bin + 1];
        const auto magnitude = std::sqrt((real * real) + (imaginary * imaginary));

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

    if (bestBin <= 0 || bestScore <= averageBandEnergy * spectralContrastThreshold)
        return updateSmoothedF0(0.0f);

    auto refinedBin = static_cast<float>(bestBin);

    const auto y1 = std::log(std::max(1.0e-12f, magnitudeSpectrum[static_cast<size_t>(bestBin - 1)]));
    const auto y2 = std::log(std::max(1.0e-12f, magnitudeSpectrum[static_cast<size_t>(bestBin)]));
    const auto y3 = std::log(std::max(1.0e-12f, magnitudeSpectrum[static_cast<size_t>(bestBin + 1)]));

    const auto denominator = y1 - (2.0f * y2) + y3;

    if (std::abs(denominator) > 1.0e-12f)
    {
        const auto delta = 0.5f * (y1 - y3) / denominator;
        refinedBin += juce::jlimit(-0.5f, 0.5f, delta);
    }

    auto candidateHz = refinedBin * binWidthHz;

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

    return updateSmoothedF0(candidateHz);
}

float ChannelSpectralSplitter::updateSmoothedF0(float candidateHz)
{
    if (candidateHz <= 0.0f)
    {
        medianWriteIndex = 0;
        medianHistoryCount = 0;

        smoothedF0Hz *= 0.9f;

        if (smoothedF0Hz < (minTrackedPitchHz * 0.25f))
            smoothedF0Hz = 0.0f;

        return smoothedF0Hz;
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
        return smoothedF0Hz;
    }

    smoothedF0Hz = (smoothingAlpha * smoothedF0Hz) + ((1.0f - smoothingAlpha) * medianHz);
    return smoothedF0Hz;
}

float ChannelSpectralSplitter::applyMedianFilter(float candidateHz)
{
    medianHistory[static_cast<size_t>(medianWriteIndex)] = candidateHz;
    medianWriteIndex = (medianWriteIndex + 1) % medianFilterLength;
    medianHistoryCount = juce::jmin(medianHistoryCount + 1, medianFilterLength);

    std::array<float, medianFilterLength> sorted{};
    std::copy_n(medianHistory.begin(), static_cast<size_t>(medianHistoryCount), sorted.begin());
    std::sort(sorted.begin(), sorted.begin() + medianHistoryCount);

    return sorted[static_cast<size_t>(medianHistoryCount / 2)];
}

void ChannelSpectralSplitter::buildMasks(float trackedF0Hz)
{
    maskA.fill(0.0f);
    maskB.fill(0.0f);
    maskN.fill(1.0f);

    if (trackedF0Hz <= 0.0f)
        return;

    const auto oddToB = juce::jmax(0.0f, harmonicsBalance);
    const auto oddToA = 1.0f - oddToB;
    const auto evenToA = juce::jmax(0.0f, -harmonicsBalance);
    const auto evenToB = 1.0f - evenToA;

    constexpr int nyquistBin = fftSize / 2;

    for (int bin = 1; bin < nyquistBin; ++bin)
    {
        const auto frequencyHz = static_cast<float>(bin) * binWidthHz;
        const auto harmonicIndex = static_cast<int>(std::lround(frequencyHz / trackedF0Hz));

        if (harmonicIndex < 1)
            continue;

        const auto centerHz = static_cast<float>(harmonicIndex) * trackedF0Hz;
        const auto distanceBins = std::abs(frequencyHz - centerHz) / binWidthHz;

        const auto baseToleranceBins = harmonicToleranceBinsBase + (harmonicToleranceGrowth * static_cast<float>(harmonicIndex));
        const auto effectiveToleranceBins = juce::jmax(0.1f, baseToleranceBins * harmonicWidth);

        if (distanceBins >= effectiveToleranceBins)
            continue;

        const auto normalizedDistance = juce::jlimit(0.0f, 1.0f, distanceBins / effectiveToleranceBins);
        const auto harmonicWeight = 0.5f * (1.0f + std::cos(juce::MathConstants<float>::pi * normalizedDistance));

        float weightA = 0.0f;
        float weightB = 0.0f;

        if ((harmonicIndex & 1) != 0)
        {
            weightA = harmonicWeight * oddToA;
            weightB = harmonicWeight * oddToB;
        }
        else
        {
            weightA = harmonicWeight * evenToA;
            weightB = harmonicWeight * evenToB;
        }

        maskA[static_cast<size_t>(bin)] = weightA;
        maskB[static_cast<size_t>(bin)] = weightB;
        maskN[static_cast<size_t>(bin)] = juce::jlimit(0.0f, 1.0f, 1.0f - (weightA + weightB));
    }
}

void ChannelSpectralSplitter::applyMasksAndReconstructFrame()
{
    harmonicsASpectrum.fill(0.0f);
    harmonicsBSpectrum.fill(0.0f);
    nonHarmonicsSpectrum.fill(0.0f);

    constexpr int nyquistBin = fftSize / 2;

    for (int bin = 0; bin <= nyquistBin; ++bin)
    {
        const auto weightA = maskA[static_cast<size_t>(bin)];
        const auto weightB = maskB[static_cast<size_t>(bin)];
        const auto weightN = maskN[static_cast<size_t>(bin)];

        if (bin == 0)
        {
            const auto source = inputSpectrum[0];

            harmonicsASpectrum[0] = source * weightA;
            harmonicsBSpectrum[0] = source * weightB;
            nonHarmonicsSpectrum[0] = source * weightN;

            continue;
        }

        if (bin == nyquistBin)
        {
            const auto source = inputSpectrum[1];

            harmonicsASpectrum[1] = source * weightA;
            harmonicsBSpectrum[1] = source * weightB;
            nonHarmonicsSpectrum[1] = source * weightN;

            continue;
        }

        const auto coefficientIndex = 2 * bin;
        const auto real = inputSpectrum[static_cast<size_t>(coefficientIndex)];
        const auto imaginary = inputSpectrum[static_cast<size_t>(coefficientIndex + 1)];

        harmonicsASpectrum[static_cast<size_t>(coefficientIndex)] = real * weightA;
        harmonicsASpectrum[static_cast<size_t>(coefficientIndex + 1)] = imaginary * weightA;
        harmonicsBSpectrum[static_cast<size_t>(coefficientIndex)] = real * weightB;
        harmonicsBSpectrum[static_cast<size_t>(coefficientIndex + 1)] = imaginary * weightB;
        nonHarmonicsSpectrum[static_cast<size_t>(coefficientIndex)] = real * weightN;
        nonHarmonicsSpectrum[static_cast<size_t>(coefficientIndex + 1)] = imaginary * weightN;
    }

    fft.performRealOnlyInverseTransform(harmonicsASpectrum.data());
    fft.performRealOnlyInverseTransform(harmonicsBSpectrum.data());
    fft.performRealOnlyInverseTransform(nonHarmonicsSpectrum.data());

    const auto requiredSize = static_cast<size_t>(nextFrameStart + fftSize);

    if (olaQueueA.size() < requiredSize)
    {
        olaQueueA.resize(requiredSize, 0.0f);
        olaQueueB.resize(requiredSize, 0.0f);
        olaQueueN.resize(requiredSize, 0.0f);
        normQueue.resize(requiredSize, 0.0f);
    }

    for (int sample = 0; sample < fftSize; ++sample)
    {
        const auto writeIndex = static_cast<size_t>(nextFrameStart + sample);
        const auto windowValue = analysisWindow[sample];

        olaQueueA[writeIndex] += harmonicsASpectrum[sample] * windowValue;
        olaQueueB[writeIndex] += harmonicsBSpectrum[sample] * windowValue;
        olaQueueN[writeIndex] += nonHarmonicsSpectrum[sample] * windowValue;
        normQueue[writeIndex] += windowSquared[sample] * ifftScale;
    }
}

void ChannelSpectralSplitter::maybePrintDebugPitch()
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

float ChannelSpectralSplitter::measureIfftScale()
{
    std::array<float, fftSize * 2> impulse{};

    impulse[0] = 1.0f;
    fft.performRealOnlyForwardTransform(impulse.data());
    fft.performRealOnlyInverseTransform(impulse.data());

    const auto measuredScale = std::abs(impulse[0]);
    return measuredScale > 1.0e-12f ? measuredScale : 1.0f;
}

HarmonicSplitAudioProcessor::HarmonicSplitAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "PARAMETERS", createParameterLayout())
{
    smoothnessParam = apvts.getRawParameterValue("smoothness");
    harmonicWidthParam = apvts.getRawParameterValue("harmonic_width");
    harmonicsBalanceParam = apvts.getRawParameterValue("harmonics_balance");
    gainAParam = apvts.getRawParameterValue("gain_a");
    gainBParam = apvts.getRawParameterValue("gain_b");
    gainNonharmParam = apvts.getRawParameterValue("gain_nonharm");
    outputModeParam = apvts.getRawParameterValue("outputMode");
}

HarmonicSplitAudioProcessor::APVTS::ParameterLayout HarmonicSplitAudioProcessor::createParameterLayout()
{
    APVTS::ParameterLayout layout;

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "smoothness", 1 },
        "Smoothness",
        juce::NormalisableRange<float>(0.0f, 100.0f, 0.1f),
        10.0f,
        juce::AudioParameterFloatAttributes().withLabel("ms")));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "harmonic_width", 1 },
        "Harmonic Width",
        juce::NormalisableRange<float>(0.1f, 2.0f, 0.01f),
        1.0f));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "harmonics_balance", 1 },
        "Harmonics Balance",
        juce::NormalisableRange<float>(-1.0f, 1.0f, 0.01f),
        0.0f));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "gain_a", 1 },
        "Gain A",
        juce::NormalisableRange<float>(-96.0f, 12.0f, 0.1f),
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "gain_b", 1 },
        "Gain B",
        juce::NormalisableRange<float>(-96.0f, 12.0f, 0.1f),
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));

    layout.add(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID { "gain_nonharm", 1 },
        "Gain Nonharm",
        juce::NormalisableRange<float>(-96.0f, 12.0f, 0.1f),
        0.0f,
        juce::AudioParameterFloatAttributes().withLabel("dB")));

    layout.add(std::make_unique<juce::AudioParameterChoice>(
        juce::ParameterID { "outputMode", 1 },
        "Output Mode",
        juce::StringArray { "All", "Harmonics A", "Harmonics B", "Non-harmonics" },
        0));

    return layout;
}

const juce::String HarmonicSplitAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool HarmonicSplitAudioProcessor::acceptsMidi() const
{
    return false;
}

bool HarmonicSplitAudioProcessor::producesMidi() const
{
    return false;
}

bool HarmonicSplitAudioProcessor::isMidiEffect() const
{
    return false;
}

double HarmonicSplitAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int HarmonicSplitAudioProcessor::getNumPrograms()
{
    return 1;
}

int HarmonicSplitAudioProcessor::getCurrentProgram()
{
    return 0;
}

void HarmonicSplitAudioProcessor::setCurrentProgram(int index)
{
    juce::ignoreUnused(index);
}

const juce::String HarmonicSplitAudioProcessor::getProgramName(int index)
{
    juce::ignoreUnused(index);
    return {};
}

void HarmonicSplitAudioProcessor::changeProgramName(int index, const juce::String& newName)
{
    juce::ignoreUnused(index, newName);
}

void HarmonicSplitAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    for (size_t channel = 0; channel < channelSplitters.size(); ++channel)
    {
        auto& splitter = channelSplitters[channel];

        splitter.prepare(sampleRate, samplesPerBlock);
        splitter.setDebugOutput(channel == 0, channel == 0 ? "F0[L]" : "F0[R]");
    }

    harmonicsATempBuffer.setSize(2, juce::jmax(1, samplesPerBlock), false, false, true);
    harmonicsBTempBuffer.setSize(2, juce::jmax(1, samplesPerBlock), false, false, true);
    nonHarmonicsTempBuffer.setSize(2, juce::jmax(1, samplesPerBlock), false, false, true);

    pitchLogFile = juce::File::getSpecialLocation(juce::File::userDesktopDirectory).getChildFile("PitchLog.txt");
    pitchLogSampleCounter = 0;

    setLatencySamples(ChannelSpectralSplitter::reportedLatencySamples);
}

void HarmonicSplitAudioProcessor::releaseResources()
{
    for (auto& splitter : channelSplitters)
        splitter.reset();

    pitchLogSampleCounter = 0;

    harmonicsATempBuffer.setSize(0, 0);
    harmonicsBTempBuffer.setSize(0, 0);
    nonHarmonicsTempBuffer.setSize(0, 0);
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool HarmonicSplitAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.inputBuses.size() != 1 || layouts.outputBuses.size() != 1)
        return false;

    if (layouts.getChannelSet(true, 0) != juce::AudioChannelSet::stereo())
        return false;

    return layouts.getChannelSet(false, 0) == juce::AudioChannelSet::stereo();
}
#endif

void HarmonicSplitAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    const auto smoothnessMs = smoothnessParam != nullptr ? smoothnessParam->load(std::memory_order_relaxed) : 10.0f;
    const auto harmonicWidth = harmonicWidthParam != nullptr ? harmonicWidthParam->load(std::memory_order_relaxed) : 1.0f;
    const auto harmonicsBalance = harmonicsBalanceParam != nullptr ? harmonicsBalanceParam->load(std::memory_order_relaxed) : 0.0f;
    const auto gainADb = gainAParam != nullptr ? gainAParam->load(std::memory_order_relaxed) : 0.0f;
    const auto gainBDb = gainBParam != nullptr ? gainBParam->load(std::memory_order_relaxed) : 0.0f;
    const auto gainNonharmDb = gainNonharmParam != nullptr ? gainNonharmParam->load(std::memory_order_relaxed) : 0.0f;
    const auto outputModeValue = outputModeParam != nullptr ? outputModeParam->load(std::memory_order_relaxed) : 0.0f;

    for (auto& splitter : channelSplitters)
        splitter.setRuntimeParameters(smoothnessMs, harmonicWidth, harmonicsBalance);

    auto inputBusBuffer = getBusBuffer(buffer, true, 0);
    auto mainOutputBus = getBusBuffer(buffer, false, 0);

    const auto numSamples = inputBusBuffer.getNumSamples();
    const auto numInputChannels = juce::jmin(2, inputBusBuffer.getNumChannels());
    const auto numOutputChannels = juce::jmin(2, mainOutputBus.getNumChannels());
    const auto numChannels = juce::jmin(numInputChannels, numOutputChannels);

    if (harmonicsATempBuffer.getNumSamples() < numSamples)
    {
        harmonicsATempBuffer.setSize(2, numSamples, false, false, true);
        harmonicsBTempBuffer.setSize(2, numSamples, false, false, true);
        nonHarmonicsTempBuffer.setSize(2, numSamples, false, false, true);
    }

    harmonicsATempBuffer.clear();
    harmonicsBTempBuffer.clear();
    nonHarmonicsTempBuffer.clear();

    for (int channel = 0; channel < numChannels; ++channel)
        channelSplitters[static_cast<size_t>(channel)].processBlock(
            inputBusBuffer.getReadPointer(channel),
            harmonicsATempBuffer.getWritePointer(channel),
            harmonicsBTempBuffer.getWritePointer(channel),
            nonHarmonicsTempBuffer.getWritePointer(channel),
            numSamples);

    harmonicsATempBuffer.applyGain(juce::Decibels::decibelsToGain(gainADb));
    harmonicsBTempBuffer.applyGain(juce::Decibels::decibelsToGain(gainBDb));
    nonHarmonicsTempBuffer.applyGain(juce::Decibels::decibelsToGain(gainNonharmDb));

    const auto selectedOutputMode = juce::jlimit(
        static_cast<int>(outputAll),
        static_cast<int>(outputNonHarmonics),
        static_cast<int>(std::lround(outputModeValue)));

    mainOutputBus.clear();

    for (int channel = 0; channel < numChannels; ++channel)
    {
        auto* destination = mainOutputBus.getWritePointer(channel);

        const auto* sourceA = harmonicsATempBuffer.getReadPointer(channel);
        const auto* sourceB = harmonicsBTempBuffer.getReadPointer(channel);
        const auto* sourceN = nonHarmonicsTempBuffer.getReadPointer(channel);

        switch (selectedOutputMode)
        {
            case outputHarmonicsA:
                std::copy_n(sourceA, numSamples, destination);
                break;

            case outputHarmonicsB:
                std::copy_n(sourceB, numSamples, destination);
                break;

            case outputNonHarmonics:
                std::copy_n(sourceN, numSamples, destination);
                break;

            case outputAll:
            default:
                for (int sample = 0; sample < numSamples; ++sample)
                    destination[sample] = sourceA[sample] + sourceB[sample] + sourceN[sample];
                break;
        }
    }

    for (int channel = numChannels; channel < mainOutputBus.getNumChannels(); ++channel)
        mainOutputBus.clear(channel, 0, numSamples);

    const auto estimatedPitch = channelSplitters[0].getSmoothedF0Hz();

    pitchLogSampleCounter += numSamples;

    if (pitchLogSampleCounter >= pitchLogIntervalSamples)
    {
        pitchLogSampleCounter %= pitchLogIntervalSamples;

        const auto logText = "Pitch: " + juce::String(estimatedPitch, 4) + " Hz\n";
        pitchLogFile.appendText(logText);
    }
}

bool HarmonicSplitAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* HarmonicSplitAudioProcessor::createEditor()
{
    return new HarmonicSplitAudioProcessorEditor(*this);
}

void HarmonicSplitAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    if (auto stateXml = apvts.copyState().createXml())
        copyXmlToBinary(*stateXml, destData);
}

void HarmonicSplitAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (auto stateXml = getXmlFromBinary(data, sizeInBytes))
        if (stateXml->hasTagName(apvts.state.getType()))
            apvts.replaceState(juce::ValueTree::fromXml(*stateXml));
}

HarmonicSplitAudioProcessor::APVTS& HarmonicSplitAudioProcessor::getAPVTS() noexcept
{
    return apvts;
}

const HarmonicSplitAudioProcessor::APVTS& HarmonicSplitAudioProcessor::getAPVTS() const noexcept
{
    return apvts;
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new HarmonicSplitAudioProcessor();
}
