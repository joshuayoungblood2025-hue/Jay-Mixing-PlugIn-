// AI Mixing Assistant — plugin editor.
//
// A compact control surface: EQ Amount / Output Gain / Ceiling sliders, a Bypass toggle,
// a "Load .chain.json" button, the loaded preset name, and simple GR read-outs.

#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class MixAssistEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit MixAssistEditor (MixAssistProcessor&);
    ~MixAssistEditor() override;

    void paint (juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;

    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;

    MixAssistProcessor& proc;

    juce::Slider eqAmount, outputGain, ceiling;
    juce::Label eqLabel, gainLabel, ceilLabel, presetLabel, meters;
    juce::ToggleButton bypass { "Bypass" };
    juce::TextButton loadButton { "Load .chain.json" };

    std::unique_ptr<SliderAttachment> eqAtt, gainAtt, ceilAtt;
    std::unique_ptr<ButtonAttachment> bypassAtt;
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MixAssistEditor)
};
