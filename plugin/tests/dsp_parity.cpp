// DSP parity harness.
//
// Includes the plugin's Dsp.h (no JUCE dependency) and reproduces exactly what
// MixAssistProcessor::processBlock does: per-channel EQ -> stereo-linked glue compression
// -> output gain -> stereo-linked limiter. It prints sampled output values so a Python
// script can confirm the C++ port matches mixassist.realtime.blocks.BusChain sample-for-
// sample. Compiles with any C++17 compiler; run by tools/verify_dsp_parity.py.

#include "../Source/Dsp.h"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace mixassist;

int main()
{
    const double sr = 48000.0;
    const int n = 2000;
    const double outGainDb = 3.0;
    const double ceilingDb = -1.0;

    // Fixed stereo test signal (must match the Python side exactly).
    std::vector<double> L(n), R(n);
    for (int i = 0; i < n; ++i)
    {
        L[i] = 0.6 * std::sin(2.0 * kPi * 220.0 * i / sr)
             + 0.3 * std::sin(2.0 * kPi * 3000.0 * i / sr);
        R[i] = 0.55 * std::sin(2.0 * kPi * 221.0 * i / sr)
             + 0.2 * std::sin(2.0 * kPi * 5000.0 * i / sr);
    }

    // EQ chain (must match the Python bands).
    std::vector<EqBandSpec> specs = {
        { EqBandSpec::Kind::HighPass,  30.0,   0.0, 0.7071 },
        { EqBandSpec::Kind::LowShelf,  100.0,  1.5, 0.7071 },
        { EqBandSpec::Kind::Peak,      300.0, -2.0, 1.0    },
        { EqBandSpec::Kind::HighShelf, 10000.0, 2.0, 0.7071 },
    };
    std::vector<Biquad> chL, chR;
    for (auto& s : specs) { chL.push_back(s.make(sr)); chR.push_back(s.make(sr)); }

    CompressorSpec cs;
    cs.enabled = true;
    cs.thresholdDb = -18.0; cs.ratio = 2.0; cs.attackMs = 30.0;
    cs.releaseMs = 200.0; cs.kneeDb = 6.0; cs.makeupDb = 0.0;
    Compressor comp; comp.prepare(cs, sr);
    Limiter lim; lim.prepare(ceilingDb, sr);

    const double outGain = std::pow(10.0, outGainDb / 20.0);

    for (int i = 0; i < n; ++i)
    {
        double l = L[i], r = R[i];
        for (auto& b : chL) l = b.processSample(l);
        for (auto& b : chR) r = b.processSample(r);

        const double peak = std::max(std::abs(l), std::abs(r));
        const double g = comp.computeGain(peak);
        l *= g; r *= g;

        l *= outGain; r *= outGain;

        const double peak2 = std::max(std::abs(l), std::abs(r));
        const double lg = lim.computeGain(peak2);
        L[i] = l * lg; R[i] = r * lg;
    }

    // Print sampled outputs (indices must match the Python comparison).
    const int idx[] = { 100, 500, 900, 1300, 1700, 1999 };
    for (int k = 0; k < 6; ++k) std::printf("%.12g ", L[idx[k]]);
    for (int k = 0; k < 6; ++k) std::printf("%.12g ", R[idx[k]]);
    std::printf("\n");
    return 0;
}
