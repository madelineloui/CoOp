import os
import pickle
import json

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

NEW_CNAMES = {
    "airplane": "airplane",
    "airport": "airport",
    "background": "background",
    "baseballfield": "baseball field",
    "basketballcourt": "basketball court",
    "bridge": "bridge",
    "chimney": "chimney",
    "dam": "dam",
    "Expressway-Service-area": "expressway service area",
    "Expressway-toll-station": "expressway toll station",
    "golffield": "golf field",
    "groundtrackfield": "ground track field",
    "harbor": "harbor",
    "overpass": "overpass",
    "ship": "ship",
    "stadium": "stadium",
    "storagetank": "storage tank",
    "tenniscourt": "tennis court",
    "trainstation": "train station",
    "vehicle": "vehicle",
    "windmill": "windmill",
}

def load_coco_split(anno_path, image_dir, new_cnames=None):
    """
    Reads a COCO annotation file and converts it into a list of Datum objects.

    Parameters:
        anno_path (str): Path to COCO annotation JSON file.
        image_dir (str): Directory where images are stored.
        new_cnames (dict): Optional mapping of class names.

    Returns:
        dataset (list): List of Datum objects (one split only).
    """
    # Load COCO JSON file
    with open(anno_path, "r") as f:
        coco_data = json.load(f)
    
    # Create mapping from category ID to class name
    cat_id_to_name = {c["id"]: c["name"] for c in coco_data["categories"]}
    
    # Apply custom class renaming if provided
    if new_cnames:
        cat_id_to_name = {cid: new_cnames[name] for cid, name in cat_id_to_name.items()}
    
    # Create a mapping from image ID to file name
    img_id_to_path = {img["id"]: os.path.join(img["folder"], img["file_name"]) for img in coco_data["images"]}
    
    # Collect all data points
    dataset = []
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        label = ann["category_id"]
        classname = cat_id_to_name[label]
        impath = img_id_to_path[image_id]
        
        #print(impath, label, classname)
        
        #dataset.append({"impath": impath, "label": label, "classname": classname})
        dataset.append(Datum(impath=impath, label=label, classname=classname))
    
    return dataset


def load_directory_split(root_dir, new_cnames=None):
    """
    Load data from a directory with class-named folders containing images.
    
    Args:
        root_dir (str): Path to the dataset root directory.
        new_cnames (dict, optional): Optional mapping from original to new class names.
    
    Returns:
        List[Datum]: A list of Datum instances.
    """
    dataset = []
    label_map = {}
    next_label = 0

    for class_folder in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        classname = new_cnames[class_folder] if new_cnames and class_folder in new_cnames else class_folder
        label = label_map.setdefault(classname, next_label)
        if label == next_label:
            next_label += 1

        for fname in os.listdir(class_path):
            if fname.lower().endswith((".jpg")):
                impath = os.path.join(class_path, fname)
                dataset.append(Datum(impath=impath, label=label, classname=classname))

    return dataset


@DATASET_REGISTRY.register()
class DIOR(DatasetBase):

    def __init__(self, cfg):
        #root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        M = cfg.SEED
        root = cfg.DATASET.ROOT
        num_shots = cfg.DATASET.NUM_SHOTS
            
        #if num_shots > 0: #TODO: don't hardcode paths
        train_path = f'/home/gridsan/manderson/ovdsat/data/cropped_data/dior/train/dior_N{num_shots}-{M}'
        val_path = f'/home/gridsan/manderson/ovdsat/data/cropped_data/dior/val/dior_val-{M}'
        test_path = f'/home/gridsan/manderson/ovdsat/data/cropped_data/dior/test/dior_test-{M}'
        # train_path = f'/home/gridsan/manderson/ovdsat/data/cropped_data/dior/train/dior_N10-1'
        # val_path = f'/home/gridsan/manderson/ovdsat/data/cropped_data/dior/val/dior_val-1'
        # test_path = f'/home/gridsan/manderson/ovdsat/data/cropped_data/dior/test/dior_test-1'
        train = load_directory_split(train_path, new_cnames=NEW_CNAMES)
        val = load_directory_split(val_path, new_cnames=NEW_CNAMES)
        test = load_directory_split(test_path, new_cnames=NEW_CNAMES)
            
            ### This is for loading from coco annotations, but doesn't crop the image so don't use anymore
#             self.dataset_dir = os.path.join(root, self.dataset_dir, 'dior')
#             self.image_dir = os.path.join(self.dataset_dir, "JPEGImages")
#             self.train_path = os.path.join(self.dataset_dir, f"train_coco_subset_N{num_shots}-{M}.json")
#             self.val_path = os.path.join(self.dataset_dir, f"train_coco_finetune_val-{M}.json")
#             self.test_path = os.path.join(self.dataset_dir, f"val_coco-{M}.json")
#             self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
            
#             mkdir_if_missing(self.split_fewshot_dir)
            
#             train = load_coco_split(self.train_path, self.image_dir, new_cnames=NEW_CNAMES)

#             print('create val subset: min(num_shots, 4)')
#             val = load_coco_split(self.val_path, self.image_dir, new_cnames=NEW_CNAMES)
#             val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4)) # Limit # val samples like original coop
#             test = load_coco_split(self.test_path, self.image_dir, new_cnames=NEW_CNAMES)

#         else:
#             dataset_dir = "/home/gridsan/manderson/ovdsat/data/dior"
#             self.dataset_dir = os.path.join(dataset_dir, 'dior')
#             self.image_dir = os.path.join(dataset_dir, "JPEGImages")
#             self.split_fewshot_dir = os.path.join(dataset_dir, "split_fewshot")
            
#             train, val, test = DTD.read_and_split_data(self.image_dir, new_cnames=NEW_CNAMES)
#             OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)
        
        print(f"Loaded training set size: {len(train) if train else 0}")
        print(f"Loaded val set size: {len(val) if val else 0}")
        print(f"Loaded test set size: {len(test) if test else 0}")

        super().__init__(train_x=train, val=val, test=test)

    def update_classname(self, dataset_old):
        dataset_new = []
        for item_old in dataset_old:
            cname_old = item_old.classname
            cname_new = NEW_CLASSNAMES[cname_old]
            item_new = Datum(impath=item_old.impath, label=item_old.label, classname=cname_new)
            dataset_new.append(item_new)
        return dataset_new
