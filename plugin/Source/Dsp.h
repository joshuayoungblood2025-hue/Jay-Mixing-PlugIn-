// DSP primitives for the AI Mixing Assistant plugin.
//
// These are direct ports of the Python engine (mixassist/dsp) so the plugin reproduces the
// exact bus chain described by a .chain.json preset: RBJ biquads, a soft-knee feed-forward
// compressor, and a feedback limiter. Header-only for simplicity.
//
// NOTE: This project is provided as source. It is designed to build on macOS with JUCE +
// Xcode/CMake and has NOT been compiled in the environment where it was generated.

#pragma once

#include <cmath>
#include <vector>

namespace mixassist
{

constexpr double kPi = 3.14159265358979323846;

// ----------------------------------------------------------------------------- Biquad

struct Biquad
{
    double b0 = 1.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
    double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;

    void reset() { x1 = x2 = y1 = y2 = 0.0; }

    inline double processSample (double x) noexcept
    {
        const double y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x; y2 = y1; y1 = y;
        return y;
    }

    static Biquad normalize (double B0, double B1, double B2, double A0, double A1, double A2)
    {
        Biquad q;
        q.b0 = B0 / A0; q.b1 = B1 / A0; q.b2 = B2 / A0;
        q.a1 = A1 / A0; q.a2 = A2 / A0;
        return q;
    }

    static Biquad lowPass (double fs, double freq, double Q = 0.7071)
    {
        const double w0 = 2.0 * kPi * freq / fs, cw = std::cos(w0), sw = std::sin(w0);
        const double alpha = sw / (2.0 * Q);
        const double b1 = 1.0 - cw, b0 = b1 / 2.0;
        return normalize(b0, b1, b0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha);
    }

    static Biquad highPass (double fs, double freq, double Q = 0.7071)
    {
        const double w0 = 2.0 * kPi * freq / fs, cw = std::cos(w0), sw = std::sin(w0);
        const double alpha = sw / (2.0 * Q);
        const double b0 = (1.0 + cw) / 2.0, b1 = -(1.0 + cw);
        return normalize(b0, b1, b0, 1.0 + alpha, -2.0 * cw, 1.0 - alpha);
    }

    static Biquad peaking (double fs, double freq, double gainDb, double Q = 1.0)
    {
        const double A = std::pow(10.0, gainDb / 40.0);
        const double w0 = 2.0 * kPi * freq / fs, cw = std::cos(w0), sw = std::sin(w0);
        const double alpha = sw / (2.0 * Q);
        return normalize(1.0 + alpha * A, -2.0 * cw, 1.0 - alpha * A,
                         1.0 + alpha / A, -2.0 * cw, 1.0 - alpha / A);
    }

    static Biquad lowShelf (double fs, double freq, double gainDb)
    {
        const double A = std::pow(10.0, gainDb / 40.0);
        const double w0 = 2.0 * kPi * freq / fs, cw = std::cos(w0), sw = std::sin(w0);
        const double alpha = sw / 2.0 * std::sqrt(2.0);
        const double t = 2.0 * std::sqrt(A) * alpha;
        return normalize(A * ((A + 1.0) - (A - 1.0) * cw + t),
                         2.0 * A * ((A - 1.0) - (A + 1.0) * cw),
                         A * ((A + 1.0) - (A - 1.0) * cw - t),
                         (A + 1.0) + (A - 1.0) * cw + t,
                         -2.0 * ((A - 1.0) + (A + 1.0) * cw),
                         (A + 1.0) + (A - 1.0) * cw - t);
    }

    static Biquad highShelf (double fs, double freq, double gainDb)
    {
        const double A = std::pow(10.0, gainDb / 40.0);
        const double w0 = 2.0 * kPi * freq / fs, cw = std::cos(w0), sw = std::sin(w0);
        const double alpha = sw / 2.0 * std::sqrt(2.0);
        const double t = 2.0 * std::sqrt(A) * alpha;
        return normalize(A * ((A + 1.0) + (A - 1.0) * cw + t),
                         -2.0 * A * ((A - 1.0) + (A + 1.0) * cw),
                         A * ((A + 1.0) + (A - 1.0) * cw - t),
                         (A + 1.0) - (A - 1.0) * cw + t,
                         2.0 * ((A - 1.0) - (A + 1.0) * cw),
                         (A + 1.0) - (A - 1.0) * cw - t);
    }
};

// ------------------------------------------------------------------------- Eq band spec

struct EqBandSpec
{
    enum class Kind { HighPass, LowPass, Peak, LowShelf, HighShelf };
    Kind kind = Kind::Peak;
    double freq = 1000.0;
    double gainDb = 0.0;
    double q = 0.7071;

    Biquad make (double fs) const
    {
        switch (kind)
        {
            case Kind::HighPass:  return Biquad::highPass(fs, freq, q);
            case Kind::LowPass:   return Biquad::lowPass(fs, freq, q);
            case Kind::LowShelf:  return Biquad::lowShelf(fs, freq, gainDb);
            case Kind::HighShelf: return Biquad::highShelf(fs, freq, gainDb);
            case Kind::Peak:
            default:              return Biquad::peaking(fs, freq, gainDb, q);
        }
    }
};

// ------------------------------------------------------------------------- Compressor

struct CompressorSpec
{
    bool enabled = false;
    double thresholdDb = -18.0;
    double ratio = 2.0;
    double attackMs = 30.0;
    double releaseMs = 200.0;
    double kneeDb = 6.0;
    double makeupDb = 0.0;
};

class Compressor
{
public:
    void prepare (const CompressorSpec& spec, double fs)
    {
        s = spec;
        atk = coef(spec.attackMs, fs);
        rel = coef(spec.releaseMs, fs);
        makeup = std::pow(10.0, spec.makeupDb / 20.0);
        env = 0.0;
    }

    // Stereo-linked, matches the Python decoupled peak detector + soft-knee curve.
    inline double computeGain (double peak) noexcept
    {
        if (peak > env) env = atk * env + (1.0 - atk) * peak;
        else            env = rel * env + (1.0 - rel) * peak;
        const double levelDb = env > 1e-12 ? 20.0 * std::log10(env) : -120.0;
        const double outDb = gainCurve(levelDb);
        return std::pow(10.0, (outDb - levelDb) / 20.0) * makeup;
    }

    bool isEnabled() const { return s.enabled; }

private:
    double gainCurve (double levelDb) const
    {
        const double thr = s.thresholdDb, ratio = s.ratio, knee = s.kneeDb;
        if (knee > 0.0 && (2.0 * (levelDb - thr)) < -knee) return levelDb;
        if (knee > 0.0 && std::abs(2.0 * (levelDb - thr)) <= knee)
        {
            const double x = levelDb - thr + knee / 2.0;
            return levelDb + (1.0 / ratio - 1.0) * (x * x) / (2.0 * knee);
        }
        return thr + (levelDb - thr) / ratio;
    }

    static double coef (double ms, double fs)
    {
        return ms <= 0.0 ? 0.0 : std::exp(-1.0 / (fs * ms / 1000.0));
    }

    CompressorSpec s;
    double atk = 0.0, rel = 0.0, makeup = 1.0, env = 0.0;
};

// ---------------------------------------------------------------------------- Limiter

class Limiter
{
public:
    void prepare (double ceilingDb, double fs, double releaseMs = 50.0)
    {
        ceiling = std::pow(10.0, ceilingDb / 20.0);
        rel = std::exp(-1.0 / (fs * releaseMs / 1000.0));
        env = 0.0;
    }

    inline double computeGain (double peak) noexcept
    {
        const double target = peak > ceiling ? ceiling / peak : 1.0;
        if (target < env || env == 0.0) env = target;      // instant attack
        else { env = rel * env + (1.0 - rel) * target; if (env > 1.0) env = 1.0; }
        return env;
    }

private:
    double ceiling = 1.0, rel = 0.0, env = 0.0;
};

} // namespace mixassist
