#pragma once

#include <JuceHeader.h>

#include <array>
#include <memory>

#include "PluginProcessor.h"

class HarmonicSplitAudioProcessorEditor final : public juce::AudioProcessorEditor
{
public:
    explicit HarmonicSplitAudioProcessorEditor(HarmonicSplitAudioProcessor&);
    ~HarmonicSplitAudioProcessorEditor() override = default;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboBoxAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;

    struct Control
    {
        juce::String title;
        juce::String parameterId;
        juce::Label label;
        juce::Slider slider;
        std::unique_ptr<SliderAttachment> attachment;
    };

    static void configureSlider(juce::Slider& slider);

    HarmonicSplitAudioProcessor& processor;

    juce::Label titleLabel;
    juce::Label outputModeLabel;
    juce::ComboBox outputModeComboBox;
    std::unique_ptr<ComboBoxAttachment> outputModeAttachment;

    std::array<Control, 6> controls;
};
