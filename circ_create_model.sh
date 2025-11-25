#!/bin/bash

export PATH="/c/Program Files/Tesseract-OCR:$PATH"
export TESSDATA_PREFIX="/c/Program Files/Tesseract-OCR/tessdata"
make training MODEL_NAME=circuit
