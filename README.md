# Infrared and Visible Image Processing and Prediction Script - inferno_web
Scripts for generating https://prosecco.psu.edu/inferno_web website pages.

This repository contains a Python script for processing infrared (IR) and visible images, generating temperature prediction graphs and thumbnails, and creating a webpage for easy visualization.

## Requirements

* Python 3.6 or higher
* PIL
* numpy
* pandas
* fnv
* matplotlib
* Jinja2

## How to Use

1. Clone this repository to your local machine.
2. Install the required packages mentioned in the requirements section.
3. Execute the script using the command below:

```bash
python script.py --visible_location /path/to/visible_files --ir_location /path/to/ir_files --output_dir /path/to/output_directory --template_dir /path/to/template_directory
```

Replace `/path/to/visible_files`, `/path/to/ir_files`, `/path/to/output_directory`, and `/path/to/template_directory` with the appropriate paths on your system.

## Features

This script processes infrared (IR) and visible images, and performs the following tasks:

* Filter images within a specified date range.
* Create directories for storing processed images.
* Create thumbnails of visible images.
* Read thermal files and generate IR thumbnails with colorbars.
* Copy and rename visible and IR files to the output directory.
* Create CSV files with temperature data from the IR images.
* Generate a webpage with the processed images.
* Update a calendar with the dates that have been processed.
* Perform temperature predictions using the PredRNN model.
* Create prediction graphs and save them as images.

## Contributing

Please feel free to create issues or submit pull requests for any improvements or bug fixes. We appreciate your help in improving the project.

## License

This project is licensed under the MIT License. Please refer to the [LICENSE](LICENSE) file for more information.
