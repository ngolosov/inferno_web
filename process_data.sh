#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate pytorch
python "/amethyst/s0/nvg5370/IR_website_script/process_data.py" "/rschnet/cervone01/www/prosecco.geog.psu.edu/htdocs/inferno_data/Visible/*.jpg" "/rschnet/cervone01/www/prosecco.geog.psu.edu/htdocs/inferno_data/IR/*.seq" "/rschnet/cervone01/www/prosecco.geog.psu.edu/htdocs/inferno_web" "/amethyst/s0/nvg5370/IR_website_script"
