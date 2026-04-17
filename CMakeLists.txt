cmake_minimum_required(VERSION 3.22)

project(HarmonicSplit VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(HARMONIC_SPLIT_USE_LOCAL_JUCE "Use local JUCE checkout" OFF)
set(JUCE_DIR "" CACHE PATH "Path to a local JUCE checkout")

if(HARMONIC_SPLIT_USE_LOCAL_JUCE)
    if(NOT EXISTS "${JUCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "JUCE_DIR must point to the JUCE repository root")
    endif()

    add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/juce)
else()
    include(FetchContent)

    FetchContent_Declare(
        juce
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 8.0.0
    )

    FetchContent_MakeAvailable(juce)
endif()

juce_add_plugin(HarmonicSplit
    COMPANY_NAME "ExampleAudio"
    BUNDLE_ID com.example.harmonicsplit
    PLUGIN_MANUFACTURER_CODE ExAu
    PLUGIN_CODE HrSp
    FORMATS VST3 AU
    PRODUCT_NAME "Harmonic Split"
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
)

target_sources(HarmonicSplit
    PRIVATE
    Source/StftOlaProcessor.cpp
    Source/StftOlaProcessor.h
        Source/PluginProcessor.cpp
        Source/PluginProcessor.h
)

target_compile_definitions(HarmonicSplit
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(HarmonicSplit
    PRIVATE
    juce::juce_dsp
        juce::juce_audio_utils
)

juce_generate_juce_header(HarmonicSplit)
