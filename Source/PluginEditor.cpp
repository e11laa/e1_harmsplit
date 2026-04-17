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
    outputModeLabel.setJustificationType(juce::Justification::centred);
    outputModeLabel.setFont(juce::Font(15.0f, juce::Font::bold));
    addAndMakeVisible(outputModeLabel);

    outputModeComboBox.addItemList(juce::StringArray { "All", "Harmonics A", "Harmonics B", "Non-harmonics" }, 1);
    outputModeComboBox.setJustificationType(juce::Justification::centred);
    outputModeComboBox.setTextWhenNothingSelected("Select Output Mode");
    outputModeComboBox.setColour(juce::ComboBox::backgroundColourId, juce::Colour(0xff1a232d));
    outputModeComboBox.setColour(juce::ComboBox::outlineColourId, juce::Colour(0xff44a9d8));
    outputModeComboBox.setColour(juce::ComboBox::textColourId, juce::Colours::white);
    outputModeComboBox.setColour(juce::ComboBox::arrowColourId, juce::Colour(0xff44a9d8));
    addAndMakeVisible(outputModeComboBox);
    outputModeAttachment = std::make_unique<ComboBoxAttachment>(processor.getAPVTS(), "outputMode", outputModeComboBox);

    const std::array<std::pair<juce::String, juce::String>, 8> controlDefinitions {
        std::pair { juce::String("Smoothness (ms)"), juce::String("smoothness") },
        std::pair { juce::String("Harmonic Width"), juce::String("harmonic_width") },
        std::pair { juce::String("Harmonics Balance"), juce::String("harmonics_balance") },
        std::pair { juce::String("Slope"), juce::String("slope") },
        std::pair { juce::String("Separation"), juce::String("separation") },
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

    setSize(600, 460);
}

void HarmonicSplitAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(0xff141619));
}

void HarmonicSplitAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(16);

    titleLabel.setBounds(bounds.removeFromTop(36));
    bounds.removeFromTop(6);

    auto modeSection = bounds.removeFromTop(58);
    outputModeLabel.setBounds(modeSection.removeFromTop(22));

    auto comboArea = modeSection.reduced(0, 2);
    outputModeComboBox.setBounds(comboArea.withSizeKeepingCentre(320, 28));

    bounds.removeFromTop(6);

    const auto columns = 3;
    const auto rows = static_cast<int>((controls.size() + static_cast<size_t>(columns - 1)) / static_cast<size_t>(columns));
    const auto cellWidth = bounds.getWidth() / columns;
    const auto cellHeight = bounds.getHeight() / juce::jmax(1, rows);

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
