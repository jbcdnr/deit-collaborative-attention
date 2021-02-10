import torchvision
from pathlib import Path
import tqdm


def main():
    tmp = Path("tmp")
    target_dir = Path("cifar10")
    dataset_train = torchvision.datasets.CIFAR10("tmp", train=True, download=True)
    dataset_test = torchvision.datasets.CIFAR10("tmp", train=True, download=True)

    for split, dataset in [("train", dataset_train), ("val", dataset_test)]:
        print(f"Processing {split} dataset")
        for i in tqdm.trange(len(dataset)):
            image, label = dataset[i]
            save_file = target_dir / split / f"{label}" / f"{i:08}.jpg"
            save_file.parent.mkdir(exist_ok=True, parents=True)
            image.save(save_file)


if __name__ == "__main__":
    main()
