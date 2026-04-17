#pragma once

#include <JuceHeader.h>
#include <juce_dsp/juce_dsp.h>

#include <array>
#include <atomic>
#include <vector>

class ChannelSpectralSplitter final
{
public:
    static constexpr int fftOrder = 11;
    static constexpr int fftSize = 1 << fftOrder;
    static constexpr int overlapFactor = 4;
    static constexpr int hopSize = fftSize / overlapFactor;
    static constexpr int reportedLatencySamples = fftSize - hopSize;

    ChannelSpectralSplitter();

    void prepare(double sampleRate, int maxBlockSize);
    void reset();

    void processBlock(const float* input,
                      float* harmonicsAOutput,
                      float* harmonicsBOutput,
                      float* nonHarmonicsOutput,
                      int numSamples);

    void setRuntimeParameters(float smoothnessMs, float harmonicWidth, float harmonicsBalance);
    void setDebugOutput(bool shouldOutput, juce::String label);
    float getSmoothedF0Hz() const noexcept;

private:
    static constexpr float minTrackedPitchHz = 40.0f;
    static constexpr float maxTrackedPitchHz = 2000.0f;
    static constexpr float silencePowerThreshold = 1.0e-7f;
    static constexpr float spectralContrastThreshold = 5.0f;
    static constexpr float harmonicToleranceBinsBase = 1.5f;
    static constexpr float harmonicToleranceGrowth = 0.02f;
    static constexpr int medianFilterLength = 5;
    static constexpr int debugPrintIntervalFrames = 24;

    void processAvailableFrames();
    void flushReadySamples();

    float estimateF0Hz(float framePower);
    float updateSmoothedF0(float candidateHz);
    float applyMedianFilter(float candidateHz);
    void buildMasks(float trackedF0Hz);
    void applyMasksAndReconstructFrame();
    void maybePrintDebugPitch();
    float measureIfftScale();

    juce::dsp::FFT fft;
    juce::dsp::WindowingFunction<float> window;

    std::array<float, fftSize> analysisWindow{};
    std::array<float, fftSize> windowSquared{};
    std::array<float, fftSize * 2> inputSpectrum{};
    std::array<float, fftSize * 2> harmonicsASpectrum{};
    std::array<float, fftSize * 2> harmonicsBSpectrum{};
    std::array<float, fftSize * 2> nonHarmonicsSpectrum{};
    std::array<float, (fftSize / 2) + 1> magnitudeSpectrum{};
    std::array<float, (fftSize / 2) + 1> maskA{};
    std::array<float, (fftSize / 2) + 1> maskB{};
    std::array<float, (fftSize / 2) + 1> maskN{};
    std::array<float, medianFilterLength> medianHistory{};

    std::vector<float> inputQueue;
    std::vector<float> olaQueueA;
    std::vector<float> olaQueueB;
    std::vector<float> olaQueueN;
    std::vector<float> normQueue;
    std::vector<float> outputQueueA;
    std::vector<float> outputQueueB;
    std::vector<float> outputQueueN;

    double sampleRateHz = 48000.0;
    float binWidthHz = 48000.0f / static_cast<float>(fftSize);
    int nextFrameStart = 0;
    int minTrackedBin = 1;
    int maxTrackedBin = (fftSize / 2) - 1;
    int medianWriteIndex = 0;
    int medianHistoryCount = 0;
    int debugFrameCounter = 0;
    float smoothnessMs = 10.0f;
    float harmonicWidth = 1.0f;
    float harmonicsBalance = 0.0f;
    float smoothingAlpha = 0.85f;
    float ifftScale = 1.0f;
    float smoothedF0Hz = 0.0f;
    bool debugOutputEnabled = false;
    juce::String debugLabel = "F0";
};

class HarmonicSplitAudioProcessor final : public juce::AudioProcessor
{
public:
    using APVTS = juce::AudioProcessorValueTreeState;

    HarmonicSplitAudioProcessor();
    ~HarmonicSplitAudioProcessor() override = default;

    static APVTS::ParameterLayout createParameterLayout();

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

#ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
#endif

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    APVTS& getAPVTS() noexcept;
    const APVTS& getAPVTS() const noexcept;

private:
    enum OutputMode : int
    {
        outputAll = 0,
        outputHarmonicsA = 1,
        outputHarmonicsB = 2,
        outputNonHarmonics = 3
    };

    static constexpr int pitchLogIntervalSamples = 5000;

    APVTS apvts;

    std::array<ChannelSpectralSplitter, 2> channelSplitters;
    juce::AudioBuffer<float> harmonicsATempBuffer;
    juce::AudioBuffer<float> harmonicsBTempBuffer;
    juce::AudioBuffer<float> nonHarmonicsTempBuffer;

    std::atomic<float>* smoothnessParam = nullptr;
    std::atomic<float>* harmonicWidthParam = nullptr;
    std::atomic<float>* harmonicsBalanceParam = nullptr;
    std::atomic<float>* gainAParam = nullptr;
    std::atomic<float>* gainBParam = nullptr;
    std::atomic<float>* gainNonharmParam = nullptr;
    std::atomic<float>* outputModeParam = nullptr;

    juce::File pitchLogFile;
    int pitchLogSampleCounter = 0;
};
