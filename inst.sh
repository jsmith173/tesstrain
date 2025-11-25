#!/usr/bin/env bash
# MSYS2 / Mingw-w64 64-bit environment setup for Tesseract + Python

# Update package database and core system
pacman -Syu

# Base development tools
pacman -S base-devel git

# Python + pip (64-bit)
pacman -S mingw-w64-x86_64-python mingw-w64-x86_64-python-pip

# Image handling libraries
pacman -S mingw-w64-x86_64-imagemagick \
          mingw-w64-x86_64-libjpeg \
          mingw-w64-x86_64-libpng

# Optional: plotting and visualization
pacman -S mingw-w64-x86_64-python-matplotlib
