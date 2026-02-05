# Usage

Usage: ./main (input_image_file) (effect_name) [contrast_min] [contrast_max]

Effect name is one of <stipple | invert | coords | sort_vert | sort_hor | contrast>

contrast_min/max are used for sort_vert, sort_hor, and contrast.

Output file is 'images/{input_file_name}_{effect_name}.png'

Pixel sorting algorithm is from Acerola (https://github.com/GarrettGunnell/Pixel-Sorting/tree/main)

# Example of pixel sorting

(source image from https://www.pexels.com/photo/man-passing-through-road-1821394/)

![non-sorted image](https://github.com/user-attachments/assets/a5ca77c2-be07-42cd-8fee-8b944a305b80)

<img width="2448" height="3264" alt="sorted image" src="https://github.com/user-attachments/assets/4e2834bc-74e2-4d6c-8191-4cfcc3f6ae5b" />

Note, the sorted image is compressed to be uploaded to github
