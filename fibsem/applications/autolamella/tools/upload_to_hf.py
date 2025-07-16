from datasets import Dataset, DatasetDict, Image, NamedSplit, DatasetInfo, ClassLabel, Features
import os
import glob


WAFFLE_TRAIN_PATH = "/home/patrick/github/data/autolamella-paper/model-development/train/autolamella-waffle/train/"
WAFFLE_TEST_PATH = "/home/patrick/github/data/autolamella-paper/model-development/train/autolamella-waffle/test/"

AUTOLIFTOUT_TRAIN_PATH = "/home/patrick/github/data/autolamella-paper/model-development/train/autoliftout/train/"
AUTOLIFTOUT_TEST_PATH = "/home/patrick/github/data/autolamella-paper/model-development/train/autoliftout/test/"

SERIAL_LIFTOUT_TRAIN_PATH = "/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/train/"
SERIAL_LIFTOUT_TEST_PATH = "/home/patrick/github/data/autolamella-paper/model-development/train/serial-liftout/test/"



def create_dataset(image_paths, label_paths, split: str, info=None):

    info = DatasetInfo(
        description="Autolamella Dataset",
        homepage= "https://github.com/DeMarcoLab/autolamella",
        license= "MIT",
        # features= Features({
        #     # "image": Image(),
        #     # "annotation": Image(),
        #     "label": ClassLabel(
        #         num_classes=6,
        #         names=["background", "lamella", "manipulator", "landing_post", "copper_adaptor", "volume_block"]),
        # }),
    )
    # TODO: add features to the dataset

    dataset = Dataset.from_dict({"image": sorted(image_paths),
                                "annotation": sorted(label_paths),
                                # "split": [split] * len(image_paths)
                                # "features": [info.features] * len(image_paths),
                                },
                                split=split,
                                # split=split,
                                # features=info.features, 
                                info=info
                                )
    dataset = dataset.cast_column("image", Image())
    dataset = dataset.cast_column("annotation", Image())


    return dataset

train_paths = [WAFFLE_TRAIN_PATH, AUTOLIFTOUT_TRAIN_PATH, SERIAL_LIFTOUT_TRAIN_PATH]
test_paths = [WAFFLE_TEST_PATH, AUTOLIFTOUT_TEST_PATH, SERIAL_LIFTOUT_TEST_PATH]


autolamella_datasets = {
    "waffle": {"train": {}, "test": {}},
    "liftout": {"train": {}, "test": {}},
    "serial-liftout": {"train": {}, "test": {}},
}

for name, train_path, test_path in zip(autolamella_datasets.keys(), train_paths, test_paths):
    # your images can of course have a different extension
    image_paths_train = sorted(glob.glob(os.path.join(train_path, "*.tif*")))
    label_paths_train = sorted(glob.glob(os.path.join(train_path, "labels", "*.tif*")))

    # same for test
    image_paths_test = sorted(glob.glob(os.path.join(test_path, "*.tif*")))
    label_paths_test = sorted(glob.glob(os.path.join(test_path, "labels", "*.tif*")))

    # step 1: create Dataset objects
    train_dataset = create_dataset(image_paths_train, label_paths_train, "train")
    test_dataset = create_dataset(image_paths_test, label_paths_test, "test")

    autolamella_datasets[name]["train"] = train_dataset
    autolamella_datasets[name]["test"] = test_dataset


# create DatasetDict
waffle_dataset = DatasetDict(autolamella_datasets["waffle"])
liftout_dataset = DatasetDict(autolamella_datasets["liftout"])
serial_liftout_dataset = DatasetDict(autolamella_datasets["serial-liftout"])


# https://huggingface.co/docs/datasets/v2.16.1/en/repository_structure



print(waffle_dataset)
print(liftout_dataset)
print(serial_liftout_dataset)


# optionally, you can push to a private repo on the hub
waffle_dataset.push_to_hub("patrickcleeve/autolamella", config_name="waffle", private=True)
liftout_dataset.push_to_hub("patrickcleeve/autolamella", config_name="liftout",private=True)
serial_liftout_dataset.push_to_hub("patrickcleeve/autolamella", config_name="serial-liftout",private=True)