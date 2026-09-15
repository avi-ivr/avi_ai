#!/usr/bin/env bash
set -e
python -m pip install -r requirements.txt
printf '%s\n' 'AVI_AI_BUILD_VERSION=2026-09-15-gemini-transcription'
printf '%s\n' 'Gemini transcription build completed successfully'
