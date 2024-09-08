# Text and Visual Elements Segmentor

## Overview

The Text and Visual Elements Segmentor is a tool designed to extract text and visual elements from images. It combines Google Cloud Vision OCR for text extraction with computer vision techniques for segmenting visual elements. The system intelligently masks text areas during contour detection to ensure accurate segmentation of visual elements.

## Features

- **Text Extraction:** Uses Google Cloud Vision OCR to extract all text from images.
- **Visual Element Segmentation:** Applies contour detection to identify and segment visual elements or shapes.
- **Text Masking:** Text areas are masked during contour detection to prevent text from being incorrectly classified as visual elements.
- **Improved Detection:** Utilizes adaptive thresholding and morphological functions to enhance contour detection accuracy.
- **User Interface:** Simple HTML-based interface for uploading images and extracting both text and visual elements.

## How It Works

1. **Text Extraction:**
   - The image is processed using Google Cloud Vision OCR to extract all text content.
   
2. **Visual Elements Segmentation:**
   - The system masks the detected text areas and performs contour detection on the remaining image.
   - Adaptive thresholding and morphological operations are applied to improve the detection of visual elements.

3. **User Interface:**
   - Users can upload images through a simple HTML interface.
   - The tool processes the image and displays the extracted text and segmented visual elements.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/text-visual-segmentor.git
   cd text-visual-segmentor
