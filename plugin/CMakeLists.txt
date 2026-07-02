cmake_minimum_required(VERSION 3.22)
project(MixAssistPlugin VERSION 0.1.0 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# JUCE. By default this fetches JUCE from GitHub. If you already have JUCE
# locally, set -DMIXASSIST_JUCE_PATH=/path/to/JUCE to use add_subdirectory
# instead (useful for offline builds).
# ---------------------------------------------------------------------------
set(MIXASSIST_JUCE_PATH "" CACHE PATH "Path to a local JUCE checkout (optional)")

if(MIXASSIST_JUCE_PATH)
    add_subdirectory(${MIXASSIST_JUCE_PATH} ${CMAKE_BINARY_DIR}/JUCE)
else()
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG        8.0.4
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(JUCE)
endif()

juce_add_plugin(MixAssist
    COMPANY_NAME              "MixAssist"
    PLUGIN_MANUFACTURER_CODE  Mxas
    PLUGIN_CODE               Mxa1
    FORMATS                   AU VST3 Standalone
    PRODUCT_NAME              "AI Mixing Assistant"
    IS_SYNTH                  FALSE
    NEEDS_MIDI_INPUT          FALSE
    NEEDS_MIDI_OUTPUT         FALSE
    COPY_PLUGIN_AFTER_BUILD   TRUE)

juce_generate_juce_header(MixAssist)

target_sources(MixAssist PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp)

target_compile_features(MixAssist PRIVATE cxx_std_17)

target_compile_definitions(MixAssist PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0)

target_link_libraries(MixAssist
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags)

# ---------------------------------------------------------------------------
# Optional: the JUCE-free DSP parity harness (proves Dsp.h matches the Python
# engine). Enable with -DMIXASSIST_BUILD_TESTS=ON.
# ---------------------------------------------------------------------------
option(MIXASSIST_BUILD_TESTS "Build the DSP parity harness" OFF)
if(MIXASSIST_BUILD_TESTS)
    add_executable(dsp_parity tests/dsp_parity.cpp)
    target_compile_features(dsp_parity PRIVATE cxx_std_17)
endif()
