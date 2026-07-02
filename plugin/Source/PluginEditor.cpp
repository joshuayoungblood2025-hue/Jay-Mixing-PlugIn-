#include "PluginEditor.h"

namespace
{
void styleRotary (juce::Slider& s)
{
    s.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    s.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 72, 18);
}
void styleCaption (juce::Label& l, const juce::String& text)
{
    l.setText (text, juce::dontSendNotification);
    l.setJustificationType (juce::Justification::centred);
    l.setFont (juce::Font (13.0f));
}
} // namespace

MixAssistEditor::MixAssistEditor (MixAssistProcessor& p)
    : juce::AudioProcessorEditor (&p), proc (p)
{
    for (auto* s : { &eqAmount, &outputGain, &ceiling })
    {
        styleRotary (*s);
        addAndMakeVisible (*s);
    }
    styleCaption (eqLabel, "EQ Amount");
    styleCaption (gainLabel, "Output Gain");
    styleCaption (ceilLabel, "Ceiling");
    for (auto* l : { &eqLabel, &gainLabel, &ceilLabel })
        addAndMakeVisible (*l);

    addAndMakeVisible (bypass);
    addAndMakeVisible (loadButton);

    presetLabel.setJustificationType (juce::Justification::centredLeft);
    presetLabel.setText ("Preset: " + proc.getLoadedPresetName(), juce::dontSendNotification);
    addAndMakeVisible (presetLabel);

    meters.setJustificationType (juce::Justification::centredLeft);
    meters.setFont (juce::Font (juce::Font::getDefaultMonospacedFontName(), 12.0f, juce::Font::plain));
    addAndMakeVisible (meters);

    eqAtt   = std::make_unique<SliderAttachment> (proc.apvts, "eqAmount", eqAmount);
    gainAtt = std::make_unique<SliderAttachment> (proc.apvts, "outputGain", outputGain);
    ceilAtt = std::make_unique<SliderAttachment> (proc.apvts, "ceiling", ceiling);
    bypassAtt = std::make_unique<ButtonAttachment> (proc.apvts, "bypass", bypass);

    loadButton.onClick = [this]
    {
        chooser = std::make_unique<juce::FileChooser> (
            "Load a .chain.json preset", juce::File(), "*.json");
        chooser->launchAsync (juce::FileBrowserComponent::openMode
                                  | juce::FileBrowserComponent::canSelectFiles,
                              [this] (const juce::FileChooser& fc)
                              {
                                  auto f = fc.getResult();
                                  if (f.existsAsFile() && proc.loadPresetFromFile (f))
                                      presetLabel.setText ("Preset: " + proc.getLoadedPresetName(),
                                                           juce::dontSendNotification);
                              });
    };

    setSize (460, 260);
    startTimerHz (15);
}

MixAssistEditor::~MixAssistEditor() { stopTimer(); }

void MixAssistEditor::paint (juce::Graphics& g)
{
    g.fillAll (juce::Colour (0xff0f172a));
    g.setColour (juce::Colour (0xffe2e8f0));
    g.setFont (juce::Font (18.0f, juce::Font::bold));
    g.drawText ("AI Mixing Assistant", 16, 12, getWidth() - 32, 24,
                juce::Justification::centredLeft);
    g.setColour (juce::Colour (0xff38bdf8));
    g.setFont (juce::Font (11.0f));
    g.drawText ("bus / master insert", 16, 34, getWidth() - 32, 16,
                juce::Justification::centredLeft);
}

void MixAssistEditor::resized()
{
    auto area = getLocalBounds().reduced (16);
    area.removeFromTop (44);

    auto knobs = area.removeFromTop (120);
    const int w = knobs.getWidth() / 3;
    auto place = [&] (juce::Slider& s, juce::Label& l, juce::Rectangle<int> r)
    {
        l.setBounds (r.removeFromBottom (18));
        s.setBounds (r);
    };
    place (eqAmount, eqLabel, knobs.removeFromLeft (w));
    place (outputGain, gainLabel, knobs.removeFromLeft (w));
    place (ceiling, ceilLabel, knobs);

    auto row = area.removeFromTop (28);
    bypass.setBounds (row.removeFromLeft (100));
    loadButton.setBounds (row.removeFromRight (160));

    presetLabel.setBounds (area.removeFromTop (22));
    meters.setBounds (area.removeFromTop (22));
}

void MixAssistEditor::timerCallback()
{
    meters.setText (juce::String::formatted ("comp GR %.1f dB    limiter GR %.1f dB",
                                              proc.compGrDb.load(), proc.limiterGrDb.load()),
                    juce::dontSendNotification);
}
