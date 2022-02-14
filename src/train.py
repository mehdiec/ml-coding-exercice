import argparse
import os
import pickle
import yaml
from shutil import copyfile
from sklearn import svm
from sklearn.model_selection import train_test_split

from data.load import train_loader


def generate_unique_logpath(logdir, raw_run_name):
    """Verify if the path already exist

    Args:
        logdir (str): path to log dir
        raw_run_name (str): name of the file

    Returns:
        str: path to the output file
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


def main(cfg, path_to_config):
    """Main pipeline to train a ML model

    Args:
        cfg (dict): config with all the necessary parameters
        path_to_config(string): path to the config file
    """
    # Load data
    cfg["DATASET"]["VALID_RATIO"]
    num_images = cfg["DATASET"]["NUM_IMAGE"]
    images, targets = train_loader(num_images=num_images)
    targets = targets[:num_images]

    train_images, test_images, train_labels, test_labels = train_test_split(
        images, targets, train_size=cfg["DATASET"]["VALID_RATIO"], random_state=0
    )

    # Init directory to save model saving best models

    top_logdir = cfg["TRAIN"]["SAVE_DIR"]
    save_dir = generate_unique_logpath(top_logdir, cfg["MODELS"]["ML"]["TYPE"].lower())
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    copyfile(path_to_config, os.path.join(save_dir, "config_file.yaml"))

    model = svm.SVC()
    model.fit(train_images, train_labels)
    pickle.dump(model, open(os.path.join(save_dir, "model.pck"), "wb"))

    print("Accuaracy : ", model.score(test_images, test_labels))


if __name__ == "__main__":
    # Init the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    # Add path to the config file to the command line arguments
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file",
    )
    args = parser.parse_args()

    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    main(cfg=config_file, path_to_config=args.path_to_config)
