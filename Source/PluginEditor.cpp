#include "PluginEditor.h"

HarmonicSplitAudioProcessorEditor::HarmonicSplitAudioProcessorEditor(HarmonicSplitAudioProcessor& processorRef)
    : AudioProcessorEditor(&processorRef),
      processor(processorRef)
{
    titleLabel.setText("Harmonic Split", juce::dontSendNotification);
    titleLabel.setJustificationType(juce::Justification::centred);
    titleLabel.setFont(juce::Font(24.0f, juce::Font::bold));
    addAndMakeVisible(titleLabel);

    outputModeLabel.setText("Output Mode", juce::dontSendNotification);
    outputModeLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(outputModeLabel);

    outputModeBox.addItemList(juce::StringArray { "All", "Harmonics A", "Harmonics B", "Non-harmonics" }, 1);
    addAndMakeVisible(outputModeBox);
    outputModeAttachment = std::make_unique<ComboBoxAttachment>(processor.getAPVTS(), "outputMode", outputModeBox);

    const std::array<std::pair<juce::String, juce::String>, 6> controlDefinitions {
        std::pair { juce::String("Smoothness (ms)"), juce::String("smoothness") },
        std::pair { juce::String("Harmonic Width"), juce::String("harmonic_width") },
        std::pair { juce::String("Harmonics Balance"), juce::String("harmonics_balance") },
        std::pair { juce::String("Gain A (dB)"), juce::String("gain_a") },
        std::pair { juce::String("Gain B (dB)"), juce::String("gain_b") },
        std::pair { juce::String("Gain Nonharm (dB)"), juce::String("gain_nonharm") }
    };

    for (size_t index = 0; index < controls.size(); ++index)
    {
        controls[index].title = controlDefinitions[index].first;
        controls[index].parameterId = controlDefinitions[index].second;
    }

    for (auto& control : controls)
    {
        control.label.setText(control.title, juce::dontSendNotification);
        control.label.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(control.label);

        configureSlider(control.slider);
        addAndMakeVisible(control.slider);

        control.attachment = std::make_unique<SliderAttachment>(processor.getAPVTS(), control.parameterId, control.slider);
    }

    setSize(600, 400);
}

void HarmonicSplitAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff141619));
}

void HarmonicSplitAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(16);

    titleLabel.setBounds(bounds.removeFromTop(36));
    bounds.removeFromTop(8);

    auto modeRow = bounds.removeFromTop(30);
    outputModeLabel.setBounds(modeRow.removeFromLeft(120));
    outputModeBox.setBounds(modeRow.removeFromLeft(220));

    bounds.removeFromTop(8);

    const auto columns = 3;
    const auto rows = 2;
    const auto cellWidth = bounds.getWidth() / columns;
    const auto cellHeight = bounds.getHeight() / rows;

    for (size_t index = 0; index < controls.size(); ++index)
    {
        const auto row = static_cast<int>(index) / columns;
        const auto column = static_cast<int>(index) % columns;

        auto cell = juce::Rectangle<int>(
            bounds.getX() + (column * cellWidth),
            bounds.getY() + (row * cellHeight),
            cellWidth,
            cellHeight).reduced(8);

        auto labelArea = cell.removeFromTop(24);
        controls[index].label.setBounds(labelArea);
        controls[index].slider.setBounds(cell);
    }
}

void HarmonicSplitAudioProcessorEditor::configureSlider(juce::Slider& slider)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 72, 20);
}
