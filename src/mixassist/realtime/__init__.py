"""Real-time / block-based processing.

This mirrors the fixed signal path a plugin runs on a bus: stateful EQ, compression and
limiting processed in arbitrary block sizes. The output is identical whether a signal is
processed in one buffer or streamed in small blocks, which is exactly the property a DAW
plugin needs. Loudness *normalization* (which needs the whole signal) is precomputed
offline into a fixed output gain carried by the preset.
"""
