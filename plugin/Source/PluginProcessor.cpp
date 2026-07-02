#include "PluginProcessor.h"
#include "PluginEditor.h"

using APVTS = juce::AudioProcessorValueTreeState;

MixAssistProcessor::MixAssistProcessor()
    : AudioProcessor (BusesProperties()
                          .withInput ("Input", juce::AudioChannelSet::stereo(), true)
                          .withOutput ("Output", juce::AudioChannelSet::stereo(), true)),
      apvts (*this, nullptr, "PARAMS", createLayout())
{
    pBypass     = apvts.getRawParameterValue ("bypass");
    pOutputGain = apvts.getRawParameterValue ("outputGain");
    pEqAmount   = apvts.getRawParameterValue ("eqAmount");
    pCeiling    = apvts.getRawParameterValue ("ceiling");

    apvts.addParameterListener ("eqAmount", this);
    apvts.addParameterListener ("ceiling", this);
}

APVTS::ParameterLayout MixAssistProcessor::createLayout()
{
    using namespace juce;
    APVTS::ParameterLayout layout;
    layout.add (std::make_unique<AudioParameterBool> (ParameterID { "bypass", 1 }, "Bypass", false));
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "outputGain", 1 }, "Output Gain",
        NormalisableRange<float> (-24.0f, 24.0f, 0.1f), 0.0f));
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "eqAmount", 1 }, "EQ Amount",
        NormalisableRange<float> (0.0f, 2.0f, 0.01f), 1.0f));
    layout.add (std::make_unique<AudioParameterFloat> (
        ParameterID { "ceiling", 1 }, "Ceiling",
        NormalisableRange<float> (-6.0f, 0.0f, 0.1f), -1.0f));
    return layout;
}

bool MixAssistProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
    const auto& out = layouts.getMainOutputChannelSet();
    if (out != juce::AudioChannelSet::mono() && out != juce::AudioChannelSet::stereo())
        return false;
    return out == layouts.getMainInputChannelSet();
}

void MixAssistProcessor::prepareToPlay (double sampleRate, int)
{
    fs = sampleRate;
    coeffsDirty = true;
    comp.prepare (compSpec, fs);
    limiter.prepare (pCeiling != nullptr ? (double) pCeiling->load() : presetCeilingDb, fs);
    rebuildCoefficients();
}

void MixAssistProcessor::parameterChanged (const juce::String& id, float)
{
    if (id == "eqAmount" || id == "ceiling")
        coeffsDirty = true;
}

void MixAssistProcessor::rebuildCoefficients()
{
    const int numCh = juce::jmax (1, getTotalNumOutputChannels());
    const float eqAmount = pEqAmount != nullptr ? pEqAmount->load() : 1.0f;

    chains.assign ((size_t) numCh, {});
    for (int ch = 0; ch < numCh; ++ch)
    {
        chains[(size_t) ch].reserve (eqSpecs.size());
        for (auto spec : eqSpecs)
        {
            // EQ Amount scales gain-type bands (leaves pass filters intact).
            if (spec.kind == mixassist::EqBandSpec::Kind::Peak
                || spec.kind == mixassist::EqBandSpec::Kind::LowShelf
                || spec.kind == mixassist::EqBandSpec::Kind::HighShelf)
                spec.gainDb *= eqAmount;
            chains[(size_t) ch].push_back (spec.make (fs));
        }
    }
    comp.prepare (compSpec, fs);
    limiter.prepare (pCeiling != nullptr ? (double) pCeiling->load() : presetCeilingDb, fs);
    coeffsDirty = false;
}

void MixAssistProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&)
{
    juce::ScopedNoDenormals noDenormals;
    const int numCh = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    if (pBypass != nullptr && pBypass->load() > 0.5f)
        return;

    if (coeffsDirty.load())
        rebuildCoefficients();

    const double outGain = std::pow (10.0, (pOutputGain != nullptr ? pOutputGain->load() : 0.0) / 20.0);
    float maxCompGr = 0.0f, maxLimGr = 0.0f;

    auto* left  = buffer.getWritePointer (0);
    auto* right = numCh > 1 ? buffer.getWritePointer (1) : nullptr;

    for (int n = 0; n < numSamples; ++n)
    {
        double l = left[n];
        double r = right != nullptr ? right[n] : l;

        // EQ (stateful per channel)
        for (auto& b : chains[0]) l = b.processSample (l);
        if (right != nullptr)
            for (auto& b : chains[chains.size() > 1 ? 1 : 0]) r = b.processSample (r);

        // Glue compression (stereo-linked)
        if (comp.isEnabled())
        {
            const double peak = juce::jmax (std::abs (l), std::abs (r));
            const double g = comp.computeGain (peak);
            l *= g; r *= g;
            const float grDb = (float) (-20.0 * std::log10 (juce::jmax (1e-9, g)));
            maxCompGr = juce::jmax (maxCompGr, grDb);
        }

        // Output gain
        l *= outGain; r *= outGain;

        // Safety limiter (stereo-linked)
        const double peak2 = juce::jmax (std::abs (l), std::abs (r));
        const double lg = limiter.computeGain (peak2);
        if (lg < 1.0)
            maxLimGr = juce::jmax (maxLimGr, (float) (-20.0 * std::log10 (juce::jmax (1e-9, lg))));

        left[n] = (float) (l * lg);
        if (right != nullptr) right[n] = (float) (r * lg);
    }

    compGrDb.store (maxCompGr);
    limiterGrDb.store (maxLimGr);
}

// ----------------------------------------------------------------------- preset loading

static mixassist::EqBandSpec::Kind kindFromString (const juce::String& s)
{
    if (s == "highpass")  return mixassist::EqBandSpec::Kind::HighPass;
    if (s == "lowpass")   return mixassist::EqBandSpec::Kind::LowPass;
    if (s == "lowshelf")  return mixassist::EqBandSpec::Kind::LowShelf;
    if (s == "highshelf") return mixassist::EqBandSpec::Kind::HighShelf;
    return mixassist::EqBandSpec::Kind::Peak;
}

bool MixAssistProcessor::loadPresetFromText (const juce::String& jsonText)
{
    auto parsed = juce::JSON::parse (jsonText);
    if (! parsed.isObject())
        return false;

    std::vector<mixassist::EqBandSpec> newEq;
    if (auto* eqArr = parsed.getProperty ("eq", {}).getArray())
    {
        for (auto& v : *eqArr)
        {
            mixassist::EqBandSpec spec;
            spec.kind   = kindFromString (v.getProperty ("kind", "peak").toString());
            spec.freq   = (double) v.getProperty ("freq", 1000.0);
            spec.gainDb = (double) v.getProperty ("gain_db", 0.0);
            spec.q      = (double) v.getProperty ("q", 0.7071);
            newEq.push_back (spec);
        }
    }

    mixassist::CompressorSpec newComp;
    auto compVar = parsed.getProperty ("comp", {});
    if (compVar.isObject())
    {
        newComp.enabled     = true;
        newComp.thresholdDb = (double) compVar.getProperty ("threshold_db", -18.0);
        newComp.ratio       = (double) compVar.getProperty ("ratio", 2.0);
        newComp.attackMs    = (double) compVar.getProperty ("attack_ms", 30.0);
        newComp.releaseMs   = (double) compVar.getProperty ("release_ms", 200.0);
        newComp.kneeDb      = (double) compVar.getProperty ("knee_db", 6.0);
        newComp.makeupDb    = (double) compVar.getProperty ("makeup_db", 0.0);
    }

    eqSpecs = std::move (newEq);
    compSpec = newComp;
    presetCeilingDb = (double) parsed.getProperty ("limiter_ceiling_db", -1.0);
    presetOutputGainDb = (double) parsed.getProperty ("output_gain_db", 0.0);
    presetName = parsed.getProperty ("name", "preset").toString();

    // Reflect preset values into the automatable params.
    if (auto* p = apvts.getParameter ("outputGain"))
        p->setValueNotifyingHost (apvts.getParameterRange ("outputGain")
                                      .convertTo0to1 ((float) presetOutputGainDb));
    if (auto* p = apvts.getParameter ("ceiling"))
        p->setValueNotifyingHost (apvts.getParameterRange ("ceiling")
                                      .convertTo0to1 ((float) presetCeilingDb));

    coeffsDirty = true;
    return true;
}

bool MixAssistProcessor::loadPresetFromFile (const juce::File& file)
{
    if (! file.existsAsFile())
        return false;
    return loadPresetFromText (file.loadFileAsString());
}

// ----------------------------------------------------------------------------- state

void MixAssistProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = apvts.copyState();
    // Persist the loaded preset so a session recalls it.
    state.setProperty ("presetName", presetName, nullptr);
    juce::var eqArr;  // rebuild a JSON-ish string of the preset for recall
    juce::DynamicObject::Ptr obj = new juce::DynamicObject();
    juce::Array<juce::var> bands;
    for (const auto& b : eqSpecs)
    {
        juce::DynamicObject::Ptr bo = new juce::DynamicObject();
        const char* k = "peak";
        switch (b.kind)
        {
            case mixassist::EqBandSpec::Kind::HighPass:  k = "highpass"; break;
            case mixassist::EqBandSpec::Kind::LowPass:   k = "lowpass"; break;
            case mixassist::EqBandSpec::Kind::LowShelf:  k = "lowshelf"; break;
            case mixassist::EqBandSpec::Kind::HighShelf: k = "highshelf"; break;
            default: break;
        }
        bo->setProperty ("kind", k);
        bo->setProperty ("freq", b.freq);
        bo->setProperty ("gain_db", b.gainDb);
        bo->setProperty ("q", b.q);
        bands.add (juce::var (bo.get()));
    }
    obj->setProperty ("name", presetName);
    obj->setProperty ("eq", bands);
    if (compSpec.enabled)
    {
        juce::DynamicObject::Ptr co = new juce::DynamicObject();
        co->setProperty ("threshold_db", compSpec.thresholdDb);
        co->setProperty ("ratio", compSpec.ratio);
        co->setProperty ("attack_ms", compSpec.attackMs);
        co->setProperty ("release_ms", compSpec.releaseMs);
        co->setProperty ("knee_db", compSpec.kneeDb);
        co->setProperty ("makeup_db", compSpec.makeupDb);
        obj->setProperty ("comp", juce::var (co.get()));
    }
    obj->setProperty ("limiter_ceiling_db", presetCeilingDb);
    obj->setProperty ("output_gain_db", presetOutputGainDb);
    state.setProperty ("presetJson", juce::JSON::toString (juce::var (obj.get())), nullptr);

    juce::MemoryOutputStream mos (destData, true);
    state.writeToStream (mos);
}

void MixAssistProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    auto tree = juce::ValueTree::readFromData (data, (size_t) sizeInBytes);
    if (! tree.isValid())
        return;
    apvts.replaceState (tree);
    if (tree.hasProperty ("presetJson"))
        loadPresetFromText (tree.getProperty ("presetJson").toString());
}

juce::AudioProcessorEditor* MixAssistProcessor::createEditor()
{
    return new MixAssistEditor (*this);
}

// This creates new instances of the plugin.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new MixAssistProcessor();
}
