from datetime import timedelta
import tqdm
from glob import glob
from PIL import Image
from shutil import copyfile
import numpy as np
import pandas as pd
import fnv
import fnv.reduce
import fnv.file
from matplotlib.pyplot import subplots, style, colorbar, axis, savefig, close, tight_layout
from matplotlib import use
from jinja2 import Template
from datetime import datetime
import os
import argparse
from core.models.model_factory import Model
from core.utils import preprocess


def test_within_range(filename, start_date):
    # getting current timestamp
    seconds_from = start_date.timestamp()
    # getting timestamp one day forward
    seconds_to = seconds_from + 60 * 60 * 24
    # getting modification date of the file
    mtime = os.path.getmtime(filename)
    if seconds_from < mtime < seconds_to:
        return True
    return False


def create_dirs(output_dir, input_date):
    input_date = input_date.strftime("%Y-%m-%d")
    # create root dir in format Y-m-d
    if not os.path.exists(os.path.join(output_dir, input_date)):
        os.makedirs(os.path.join(output_dir, input_date))

    dir_names = ("visible", "visible_thumb", "IR", "IR_thumb", "IR_csv", "predictions_lstm")

    for dir_name in dir_names:
        # create IR_thumbnail dir
        dir_path = os.path.join(output_dir, input_date, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)


def vis_thumbnail(input_file, output_file):
    """ Create visible image thumbnail """
    image = Image.open(input_file)
    image.thumbnail((320, 240))
    image.save(output_file)


def read_thermal_file(filename):
    im = fnv.file.ImagerFile(filename)
    im.unit = fnv.Unit.TEMPERATURE_FACTORY
    im.temp_type = fnv.TempType.CELSIUS
    im.get_frame(0)
    data = np.array(im.final, copy=False).reshape((im.height, im.width))[::2, ::2]
    return data


def create_thermal_thumbnail(data, filepath, min_temp, max_temp):
    fig, ax = subplots(1, 1, figsize=(4, 3), dpi=100)
    img = ax.imshow(data, vmin=min_temp, vmax=max_temp, cmap="inferno")
    axis("off")
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    colorbar(img, cax=cax)
    savefig(filepath, bbox_inches='tight')
    close(fig)


def prepare_visible_files(input_files, start_date, output_dir):
    """ Copy visible files to the folder and create thumbnails"""
    for file in input_files:
        modification_date_as_string = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d_%H-%M")
        # copy and rename the files
        copyfile(file, os.path.join(output_dir, start_date.strftime("%Y-%m-%d"), "visible",
                                    modification_date_as_string + ".jpg"))
        # print(os.path.join(output_dir, modification_date_as_string+".jpg"))
        vis_thumbnail(file, os.path.join(output_dir, start_date.strftime("%Y-%m-%d"), "visible_thumb",
                                         modification_date_as_string + "_thumb.jpg"))


def prepare_ir_files(input_files, start_date, output_dir):
    files = [np.average(read_thermal_file(file)) for file in input_files]
    min_temp = np.min(files)
    max_temp = np.max(files)

    for file in input_files:
        modification_date_as_string = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d_%H-%M")
        copyfile(file,
                 os.path.join(output_dir, start_date.strftime("%Y-%m-%d"), "IR", modification_date_as_string + ".seq"))
        # read the seq file using a separate function, that returns numpy.
        data = read_thermal_file(file)
        # create a thumbnail with colorbar, using a separate function
        create_thermal_thumbnail(data, os.path.join(output_dir, start_date.strftime("%Y-%m-%d"), "IR_thumb",
                                                    modification_date_as_string + ".jpg"), min_temp, max_temp)
        create_csv_file(data, os.path.join(output_dir, start_date.strftime("%Y-%m-%d"), "IR_csv",
                                           modification_date_as_string + ".csv"))
        # copy a seq file with a right name


def create_web_page(start_date, output_dir, template_dir):
    # read visible and ir dirs
    # Zip a list of two of these items
    # populate a template?

    date_text = start_date.strftime("%Y-%m-%d")
    prev_date = start_date - timedelta(days=1)
    next_date = start_date + timedelta(days=1)

    folder_name = os.path.join(output_dir, date_text)

    output_html_path = os.path.join(folder_name, "index.shtml")

    ir_file_list = glob(os.path.join(folder_name, "IR", '*'))
    ir_file_list = [os.path.basename(x) for x in ir_file_list]

    visible_file_list = glob(os.path.join(folder_name, "visible", '*'))
    visible_file_list = [os.path.basename(x) for x in visible_file_list]

    image_list = list(zip(ir_file_list, visible_file_list))

    with open(os.path.join(template_dir, 'subfolder.tpl')) as f:
        rendered = Template(f.read()).render(date=date_text, prev_date=prev_date.strftime("%Y-%m-%d"),
                                             next_date=next_date.strftime("%Y-%m-%d"), images=image_list)

    with open(output_html_path, "w") as output_html:
        output_html.write(rendered)


def create_csv_file(data, filepath):
    np.savetxt(filepath, data, delimiter=",", fmt="%1.2f")


def update_calendar(output_dir, template_dir):
    output_html_path = os.path.join(output_dir, "dates.js")

    output_dirs = [x.split("-") for x in next(os.walk(output_dir))[1]]
    with open(os.path.join(template_dir, 'dates.tpl')) as f:
        rendered = Template(f.read()).render(dates=output_dirs)

    with open(output_html_path, "w") as output_html:
        output_html.write(rendered)


class Args:
    """" This class simulates the input arguments read by the original Tsihghua univ. implementation of PredRNN"""

    def __init__(self):
        self.img_width = 128
        self.total_length = 48
        self.input_length = 24
        self.num_hidden = "128, 128, 128, 128"
        self.model_name = 'predrnn_v2'
        self.visual = 0
        self.visual_path = ''
        self.patch_size = 4
        self.img_channel = 1
        self.filter_size = 5
        self.stride = 1
        self.layer_norm = 0
        self.device = 'cuda'
        self.lr = 0.0001
        self.reverse_scheduled_sampling = 1
        self.batch_size = 1
        self.decouple_beta = 0.1


def denormalize(input_value):
    """ This helper function scales back the input values between 0..1 to the original Celsius values"""
    data_min = -4.9491  # these values are min and max of the training dataset for RNN
    data_max = 52.5941
    return input_value * (data_max - data_min) + data_min


def predict_predrnn(model, input_images, configs, output_folder):
    """This function predicts the sequence of images and saves them as CSV files in the output folder"""
    # reverse schedule sampling
    if configs.reverse_scheduled_sampling == 1:
        mask_input = 1
    else:
        mask_input = configs.input_length

    real_input_flag = np.zeros(
        (configs.batch_size,
         configs.total_length - mask_input - 1,
         configs.img_width // configs.patch_size,
         configs.img_width // configs.patch_size,
         configs.patch_size ** 2 * configs.img_channel))

    if configs.reverse_scheduled_sampling == 1:
        real_input_flag[:, :configs.input_length - 1, :, :] = 1.0

    test_dat = preprocess.reshape_patch(input_images, configs.patch_size)
    try:
        # prediction happens here ???
        img_gen = model.test(test_dat, real_input_flag)

        img_gen = preprocess.reshape_patch_back(img_gen, configs.patch_size)
        output_length = configs.total_length - configs.input_length
        img_out = img_gen[:, -output_length:]
        # save prediction examples
        for i in range(output_length):
            file_name = os.path.join(output_folder, f"pred_{i + 25}.csv")
            # scaling from 0..1 to temperatures in C
            img_pd = denormalize(img_out[0, i, :, :, :])
            # save predicted images as numpy arrays
            np.savetxt(file_name, np.squeeze(img_pd))
    except Exception as e:
        print(e)


def read_seq_from_directory(source_dir):
    """ This function reads FLIR *.seq files from the specified folder and returns a normalized array to run
    predictions using the PredRNN
    """
    # need to normalize between 0..1
    min_k = 268.2009
    max_k = 325.7441
    # found on Jan 10 2023 - glob does not always return files in order. Need to sort them
    filedir = sorted(glob(source_dir))
    dataset = []
    mean_t = []

    # read every 6th file
    for file in filedir[::6]:
        im = fnv.file.ImagerFile(file)
        im.unit = fnv.Unit.TEMPERATURE_FACTORY
        im.temp_type = fnv.TempType.KELVIN
        im.get_frame(0)
        data = np.array(im.final, copy=False).reshape((im.height, im.width))[128:128 + 128, 310:310 + 128]
        dataset.append(data)
        mean_t.append(np.mean(data) - 273.15)

    dataset = dataset * 2

    dataset_array = np.array(dataset)
    dataset_array = np.expand_dims(dataset_array, axis=0)
    dataset_array = np.expand_dims(dataset_array, axis=4)
    dataset_array = (dataset_array - min_k) / (max_k - min_k)
    return dataset_array, mean_t


def create_prediction_graphs(input_temperatures, prediction_files_path):
    input_vals = []
    ground_truth_mean = []
    predictions_mean = []
    prediction_images = []
    try:
        for i in range(0, 49):
            if i < 24:
                # Populating INPUT array
                input_vals.append(input_temperatures[i])
                predictions_mean.append(None)
                ground_truth_mean.append(None)
            if i > 24:
                # populating PREDICTION array
                input_vals.append(None)
                prediction = np.genfromtxt(f"{prediction_files_path}/pred_{i}.csv")
                predictions_mean.append(np.mean(prediction))
                prediction_images.append(prediction)

        predictions_mean.insert(24, None)
        input_vals.insert(0, None)

        ax = pd.DataFrame({'Today mean t°': input_vals, 'Predicted next-day t°': predictions_mean}).plot()
        ax.set_xlabel('Today                                         Next day\n Time (hours)')
        ax.set_ylabel('Mean temperature °C')

        positions = range(0, 49)
        labels = [x if x % 3 == 0 else '' for x in list(range(0, 24)) + list(range(0, 25))]
        # labels[24] = ''
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        # gray dashed vertical line
        ax.axvline(x=24, ymin=0, ymax=1, linestyle='--', color='gray')
        tight_layout()
        savefig(f'{prediction_files_path}/predictions_graph.png')

        fig, axes = subplots(6, 4, figsize=(8, 12))

        vmin = np.min(prediction_images)
        vmax = np.max(prediction_images)

        for idx, ax in enumerate(axes.flat):
            # Ground truth images column
            img = ax.imshow(np.squeeze(prediction_images[idx]), cmap="inferno", vmin=vmin, vmax=vmax)
            ax.set_title(f"{idx + 1}:00")
            ax.axis("off")

        tight_layout()
        savefig(f'{prediction_files_path}/predictions_images.png')
    except Exception as e:
        print(e)


def create_lstm_predictions(input_date, out_dir, model, args):
    start_date = input_date.strftime("%Y-%m-%d")
    source_dir = os.path.join(out_dir, start_date, "IR", "*.seq")
    output_dir = os.path.join(out_dir, start_date, "predictions_lstm")

    # read source FLIR files for the specific day
    source_seq, input_temperatures = read_seq_from_directory(source_dir)
    # create predictions and save them to the output folder
    predict_predrnn(model, source_seq, args, output_dir)
    create_prediction_graphs(input_temperatures, output_dir)


def initialize_lstm_model():
    model_path = '/amethyst/s0/nvg5370/IR_website_script/lstm_model24hrs.pytorch'
    args = Args()
    model = Model(args)
    model.load(model_path)
    return model, args


def main():
    use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument("visible_location", help="Visible files location")
    parser.add_argument("ir_location", help="IR files location")
    parser.add_argument("output_dir", help="Output dir location")
    parser.add_argument("template_dir", help="TPL file location")

    args = parser.parse_args()

    # creating a range of dates to process
    try:
        last_output_dir = sorted([x for x in next(os.walk(args.output_dir))[1]])[-1]
    except IndexError:
        last_output_dir = "2022-07-15"

    start = datetime.strptime(last_output_dir, "%Y-%m-%d")
    end = datetime.today()
    date_generated = [start + timedelta(days=x) for x in range(0, (end - start).days)]

    # initialize LSTM
    model, lstm_args = initialize_lstm_model()

    # iterate over the generated dates
    for start_date in tqdm.tqdm(date_generated):
        visible_files = [mfile for mfile in glob(args.visible_location) if test_within_range(mfile, start_date)]
        ir_files = [mfile for mfile in glob(args.ir_location) if test_within_range(mfile, start_date)]

        create_dirs(args.output_dir, start_date)
        prepare_visible_files(visible_files, start_date, args.output_dir)
        prepare_ir_files(ir_files, start_date, args.output_dir)
        create_lstm_predictions(start_date, args.output_dir, model, lstm_args)
        create_web_page(start_date, args.output_dir, args.template_dir)

    update_calendar(args.output_dir, args.template_dir)


if __name__ == "__main__":
    main()
