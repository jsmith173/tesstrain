#!/bin/bash

export PATH="/c/Program Files/Tesseract-OCR:$PATH"
export TESSDATA_PREFIX="/c/Program Files/Tesseract-OCR/tessdata"
make training MODEL_NAME=ocrd START_MODEL=deu_latf TESSDATA=../tessdata MAX_ITERATIONS=500 LEARNING_RATE=0.001
