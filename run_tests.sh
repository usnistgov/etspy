#!/bin/bash
# 
# This file can be used to run the ETSpy test suite, and it will
# auto-select whether or not to include the GPU/CUDA code in the
# coverage profile based on whether or not CUDA is available.

CUDA=$(python -c "import astra; print(1 if astra.astra.use_cuda() else 0)")
PYTEST="pytest"

if ((CUDA==1)); then
    COVERAGE_RCFILE="etspy/tests/.coveragerc-cuda" $PYTEST
else
    COVERAGE_RCFILE="etspy/tests/.coveragerc-nocuda" $PYTEST
fi

