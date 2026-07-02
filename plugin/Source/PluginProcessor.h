// AI Mixing Assistant — plugin processor.
//
// A bus/master insert that applies the fixed chain described by a .chain.json preset
// (exported by `mixassist preset` or snapshotted from a mix): parametric EQ -> glue
// compression -> output gain -> safety limiter. The DSP matches the Python engine so a
// plugin instance reproduces the offline result on the same material.

#pragma once

#include <JuceHeader.h>
#include <atomic>
#include <vector>
#include "Dsp.h"

class MixAssistProcessor : public juce::AudioProcessor,
                           private juce::AudioProcessorValueTreeState::Listener
{
public:
    MixAssistProcessor();
    ~MixAssistProcessor() override = default;

    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;
    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "AI Mixing Assistant"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}

    void getStateInformation (juce::MemoryBlock&) override;
    void setStateInformation (const void*, int sizeInBytes) override;

    // Load a .chain.json preset produced by the Python tool. Returns true on success.
    bool loadPresetFromFile (const juce::File& file);
    bool loadPresetFromText (const juce::String& jsonText);
    juce::String getLoadedPresetName() const { return presetName; }

    juce::AudioProcessorValueTreeState apvts;

    // Meters for the editor (read-only).
    std::atomic<float> compGrDb { 0.0f };
    std::atomic<float> limiterGrDb { 0.0f };

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void parameterChanged (const juce::String& id, float newValue) override;
    void rebuildCoefficients();

    double fs = 44100.0;
    std::atomic<bool> coeffsDirty { true };

    std::vector<mixassist::EqBandSpec> eqSpecs;
    mixassist::CompressorSpec compSpec;
    double presetCeilingDb = -1.0;
    double presetOutputGainDb = 0.0;
    juce::String presetName { "(none — pass-through until a preset is loaded)" };

    // Per-channel realized state.
    std::vector<std::vector<mixassist::Biquad>> chains;   // [channel][band]
    mixassist::Compressor comp;
    mixassist::Limiter limiter;

    std::atomic<float>* pBypass = nullptr;
    std::atomic<float>* pOutputGain = nullptr;
    std::atomic<float>* pEqAmount = nullptr;
    std::atomic<float>* pCeiling = nullptr;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MixAssistProcessor)
};
