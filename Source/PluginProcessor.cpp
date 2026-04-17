#include "PluginProcessor.h"

HarmonicSplitAudioProcessor::HarmonicSplitAudioProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Harmonics A", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Harmonics B", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Non-harmonics", juce::AudioChannelSet::stereo(), true))
{
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
    for (size_t channel = 0; channel < stftProcessors.size(); ++channel)
    {
        auto& processor = stftProcessors[channel];

        processor.prepare(sampleRate, samplesPerBlock);
        processor.setDebugOutput(channel == 0, channel == 0 ? "F0[L]" : "F0[R]");
    }

    reconstructedBuffer.setSize(2, juce::jmax(1, samplesPerBlock), false, false, true);

    setLatencySamples(StftOlaProcessor::reportedLatencySamples);
}

void HarmonicSplitAudioProcessor::releaseResources()
{
    for (auto& processor : stftProcessors)
        processor.reset();

    reconstructedBuffer.setSize(0, 0);
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool HarmonicSplitAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    if (layouts.inputBuses.size() != 1 || layouts.outputBuses.size() != 3)
        return false;

    if (layouts.getChannelSet(true, 0) != juce::AudioChannelSet::stereo())
        return false;

    for (int busIndex = 0; busIndex < 3; ++busIndex)
        if (layouts.getChannelSet(false, busIndex) != juce::AudioChannelSet::stereo())
            return false;

    return true;
}
#endif

void HarmonicSplitAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

    auto inputBusBuffer = getBusBuffer(buffer, true, 0);
    auto harmonicsABuffer = getBusBuffer(buffer, false, 0);
    auto harmonicsBBuffer = getBusBuffer(buffer, false, 1);
    auto nonHarmonicsBuffer = getBusBuffer(buffer, false, 2);

    const auto numSamples = inputBusBuffer.getNumSamples();
    const auto numChannels = juce::jmin(2, inputBusBuffer.getNumChannels());

    if (reconstructedBuffer.getNumSamples() < numSamples)
        reconstructedBuffer.setSize(2, numSamples, false, false, true);

    reconstructedBuffer.clear();

    for (int channel = 0; channel < numChannels; ++channel)
        stftProcessors[static_cast<size_t>(channel)].processBlock(
            inputBusBuffer.getReadPointer(channel),
            reconstructedBuffer.getWritePointer(channel),
            numSamples);

    auto copyToBus = [&](juce::AudioBuffer<float>& destination)
    {
        destination.clear();

        const auto channelsToCopy = juce::jmin(2, destination.getNumChannels(), reconstructedBuffer.getNumChannels());

        for (int channel = 0; channel < channelsToCopy; ++channel)
            destination.copyFrom(channel, 0, reconstructedBuffer, channel, 0, numSamples);
    };

    copyToBus(harmonicsABuffer);
    copyToBus(harmonicsBBuffer);
    copyToBus(nonHarmonicsBuffer);
}

bool HarmonicSplitAudioProcessor::hasEditor() const
{
    return true;
}

juce::AudioProcessorEditor* HarmonicSplitAudioProcessor::createEditor()
{
    return new juce::GenericAudioProcessorEditor(*this);
}

void HarmonicSplitAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    juce::ignoreUnused(destData);
}

void HarmonicSplitAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    juce::ignoreUnused(data, sizeInBytes);
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new HarmonicSplitAudioProcessor();
}
