import os
import matplotlib.pyplot as plt

def count_classes_distribution():
    dataset_dir = "dataset_yolo_downsampled"
    # count classes
    count_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }
    for split in ["train", "val"]:
        if split == "val":
            continue
        split_dir = os.path.join(dataset_dir, "labels", split)
        print(f"Processing {split_dir}...")
        for fname in os.listdir(split_dir):
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(split_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id in count_map:
                        count_map[cls_id] += 1

    print("Class distribution:")
    for cls_id, count in count_map.items():
        print(f"Class {cls_id}: {count} instances")
    # plot distribution
    classes = list(count_map.keys())
    counts = [count_map[c] for c in classes]
    plt.bar(classes, counts)
    plt.xlabel("Class ID")
    plt.ylabel("Number of Instances")
    plt.title("Class Distribution in Dataset")
    plt.xticks(classes)
    plt.savefig("class_distribution.png")
    plt.show()

def create_balanced_dataset():
    # downsampling
    dataset_dir = "dataset_yolo"
    balanced_dataset_dir = "dataset_yolo_downsampled"
    os.makedirs(balanced_dataset_dir, exist_ok=True)
    total_count_map = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
    }
    # let class 0 become 6000 instances
    for split in ["train", "val"]:
        split_dir = os.path.join(dataset_dir, "labels", split)
        os.makedirs(os.path.join(balanced_dataset_dir, "labels", split), exist_ok=True)
        os.makedirs(os.path.join(balanced_dataset_dir, "images", split), exist_ok=True)
        print(f"Processing {split_dir}...")
        for fname in os.listdir(split_dir):
            count_map = {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
            }
            if not fname.endswith(".txt"):
                continue
            fpath = os.path.join(split_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cls_id = int(parts[0])
                    if cls_id in count_map:
                        count_map[cls_id] += 1
            # decide whether to copy this file
            copy_file = True
            if count_map[1] == 0 and count_map[2] == 0 and count_map[3] == 0 and split == "train":
                # only class 0
                copy_file = False
            if copy_file:
                total_count_map[0] += count_map[0]
                total_count_map[1] += count_map[1]
                total_count_map[2] += count_map[2]
                total_count_map[3] += count_map[3]
                # copy image file
                src_img_path = os.path.join(dataset_dir, "images", split, fname.replace(".txt", ".png"))
                dst_img_path = os.path.join(balanced_dataset_dir, "images", split, fname.replace(".txt", ".png"))
                os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
                os.system(f"cp {src_img_path} {dst_img_path}")

    print("Class distribution:")
    for cls_id, count in total_count_map.items():
        print(f"Class {cls_id}: {count} instances")
    # plot distribution
    classes = list(total_count_map.keys())
    counts = [total_count_map[c] for c in classes]
    plt.bar(classes, counts)
    plt.xlabel("Class ID")
    plt.ylabel("Number of Instances")
    plt.title("Class Distribution in Dataset")
    plt.xticks(classes)
    plt.savefig("class_distribution.png")
    plt.show()

if __name__ == "__main__":
    count_classes_distribution()