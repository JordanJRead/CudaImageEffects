# Usage

Usage: ./main (input_image_file) (effect_name) [contrast_min] [contrast_max]

Effect name is one of <stipple | invert | coords | sort_vert | sort_hor | contrast>

contrast_min/max are used for sort_vert, sort_hor, and contrast.

Output file is 'images/{input_file_name}_{effect_name}.png'

Pixel sorting algorithm is from Acerola (https://github.com/GarrettGunnell/Pixel-Sorting/tree/main)