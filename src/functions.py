import json

import cv2
import numpy as np
import PIL
import torch
import torchvision
import os
import xml.etree.ElementTree as ET
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch_snippets import Report

path = os.path.abspath(__file__).split('\\')
path.pop(len(path) - 1)
base_dir = '\\'.join(path)
path.pop(len(path) - 1)
base_dir_for_eval = '\\'.join(path)


# Classes for training
label_dict = {"mastercard": 0, 'mir': 1, 'union pay': 2, 'visa': 3, 'sber': 4, 'urfu': 5}
reverse_label_dict = {0: "mastercard", 1: 'mir', 2: 'union pay', 3: 'visa', 4: 'sber', 5: 'urfu'}

# Connect to the GPU if one exists.
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def collate_fn(batch):
    return tuple(zip(*batch))


def xml_to_dict(xml_path):
    # Decode the .xml file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    return {"filename": xml_path,
            "image_width": int(root.find("./size/width").text),
            "image_height": int(root.find("./size/height").text),
            "image_channels": int(root.find("./size/depth").text),
            "label": root.find("./object/name").text,
            "x1": int(root.find("./object/bndbox/xmin").text),
            "y1": int(root.find("./object/bndbox/ymin").text),
            "x2": int(root.find("./object/bndbox/xmax").text),
            "y2": int(root.find("./object/bndbox/ymax").text)}


class VisaLogoDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, annotations_path, root, transforms=None):
        self.image_path = image_path
        self.ann_path = annotations_path
        self.root = root
        self.transforms = transforms
        self.files = sorted(os.listdir(self.image_path))
        for i in range(len(self.files)):
            self.files[i] = self.files[i].split(".")[0]
            self.label_dict = label_dict

    def __getitem__(self, i):
        img = PIL.Image.open(os.path.join(self.root,
                                          f"{self.image_path}\\" + self.files[i] + ".jpg")).convert("RGB")
        ann = xml_to_dict(os.path.join(self.root,
                                       f"{self.ann_path}\\" + self.files[i] + ".xml"))
        target = {}
        target["boxes"] = torch.as_tensor([[ann["x1"],
                                            ann["y1"],
                                            ann["x2"],
                                            ann["y2"]]],
                                          dtype=torch.float32)
        target["labels"] = torch.as_tensor([label_dict[ann["label"]]],
                                           dtype=torch.int64)
        target["image_id"] = torch.as_tensor(i)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.files)


class Compose:
    def __init__(self, transforms=[]):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(torch.nn.Module):
    def forward(self, image, target=None):
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image, target=None):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                width, _ = F.get_image_size(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target


def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
    if train == True:
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def create_trainig_data(dataset_path):
    images = rf'{dataset_path}\images'
    annotations = rf'{dataset_path}\annotations'
    train_ds = VisaLogoDataset(images, annotations, ".\\", get_transform(train=True))
    val_ds = VisaLogoDataset(images, annotations, ".\\", get_transform(train=False))

    indices = torch.randperm(len(train_ds)).tolist()
    train_ds = torch.utils.data.Subset(train_ds,
                                       indices[:int(len(indices) * 0.64)])
    val_ds = torch.utils.data.Subset(val_ds,
                                     indices[int(len(indices) * 0.64):int(len(indices) * 0.8)])

    train_dl = torch.utils.data.DataLoader(train_ds,
                                           batch_size=4,
                                           shuffle=True,
                                           collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(val_ds,
                                         batch_size=4,
                                         shuffle=False,
                                         collate_fn=collate_fn)

    return train_dl, val_dl


def get_object_detection_model(num_classes=len(label_dict),
                               feature_extraction=True):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    if feature_extraction == True:
        for p in model.parameters():
            p.requires_grad = False
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
                                                      num_classes)
    return model


def unbatch(batch, device):
    X, y = batch
    X = [x.to(device) for x in X]
    y = [{k: v.to(device) for k, v in t.items()} for t in y]
    return X, y


def train_batch(batch, model, optimizer, device):
    model.train()
    X, y = unbatch(batch, device=device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    loss.backward()
    optimizer.step()
    return loss, losses


@torch.no_grad()
def validate_batch(batch, model, optimizer, device):
    model.train()
    X, y = unbatch(batch, device=device)
    optimizer.zero_grad()
    losses = model(X, y)
    loss = sum(loss for loss in losses.values())
    return loss, losses


def train_fasterrcnn(model,
                     optimizer,
                     n_epochs,
                     train_loader,
                     test_loader=None,
                     log=None,
                     keys=None,
                     device="cpu"):
    if log is None:
        log = Report(n_epochs)
    if keys is None:
        keys = ["loss_classifier",
                "loss_box_reg",
                "loss_objectness",
                "loss_rpn_box_reg"]
    model.to(device)
    for epoch in range(n_epochs):
        N = len(train_loader)
        for ix, batch in enumerate(train_loader):
            loss, losses = train_batch(batch, model,
                                       optimizer, device)
            pos = epoch + (ix + 1) / N
            log.record(pos=pos, trn_loss=loss.item(),
                       end="\r")
        if test_loader is not None:
            N = len(test_loader)
            for ix, batch in enumerate(test_loader):
                loss, losses = validate_batch(batch, model,
                                              optimizer, device)
                pos = epoch + (ix + 1) / N
                log.record(pos=pos, val_loss=loss.item(),
                           end="\r")
    log.report_avgs(epoch + 1)
    return log


def train_model(dataset_path, epochs_num, weights_name):
    model = get_object_detection_model()
    train_dl, val_dl = create_trainig_data(dataset_path)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=0.005,
                                momentum=0.9,
                                weight_decay=0.0005)
    log = train_fasterrcnn(model=model,
                           optimizer=optimizer,
                           n_epochs=epochs_num,
                           train_loader=train_dl,
                           test_loader=val_dl,
                           log=None, keys=None,
                           device=device)

    torch.save(model.state_dict(), f"{base_dir_for_eval}\\{weights_name}.pth")


@torch.no_grad()
def predict_batch(batch, model, device):
    model.to(device)
    model.eval()
    X, _ = unbatch(batch, device=device)
    predictions = model(X)
    return [x.cpu() for x in X], predictions


def decode_prediction(prediction, score_threshold=0.8, nms_iou_threshold=0.2):
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]
    if score_threshold is not None:
        want = scores > score_threshold
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]
    if nms_iou_threshold is not None:
        want = torchvision.ops.nms(boxes=boxes, scores=scores,
                                   iou_threshold=nms_iou_threshold)
        boxes = boxes[want]
        scores = scores[want]
        labels = labels[want]
    return (boxes.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy())


def compile_model(weights_path):
    model = get_object_detection_model(num_classes=len(label_dict),
                                       feature_extraction=False)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def prepro_img(image_path):
    image = cv2.imread(image_path)
    orig_image = image.copy()
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float)
    image = torch.unsqueeze(image, 0)
    return image, orig_image


def predict(model_compiled, image, orig_image, detection_threshold=0.8):
    with torch.no_grad():
        outputs = model_compiled(image)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        pred_classes = ["visa" for i in outputs[0]['labels'].cpu().numpy()]

        for j, box in enumerate(draw_boxes):
            cv2.rectangle(orig_image,
                          (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])),
                          (0, 0, 255), 2)
            cv2.putText(orig_image, pred_classes[j],
                        (int(box[0]), int(box[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                        2, lineType=cv2.LINE_AA)
        cv2.imshow('Prediction', orig_image)
        cv2.waitKey()


def predict_on_batch(model_compiled, test_dataset_path, detection_threshold=0.5):
    images_dict = {}
    for fn in os.listdir(rf"{test_dataset_path}\images"):
        im, oim = prepro_img(rf'{test_dataset_path}\images\{fn}')
        images_dict[im] = oim
    with torch.no_grad():
        for img in images_dict.keys():
            orig_image = images_dict[img]
            outputs = model_compiled(img)
            print(outputs)
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            if len(outputs[0]['boxes']) != 0:
                boxes = outputs[0]['boxes'].data.numpy()
                scores = outputs[0]['scores'].data.numpy()
                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                draw_boxes = boxes.copy()
                pred_classes = ['x' for i in outputs[0]['labels'].cpu().numpy()]
                for j, box in enumerate(draw_boxes):
                    cv2.rectangle(orig_image,
                                  (int(box[0]), int(box[1])),
                                  (int(box[2]), int(box[3])),
                                  (0, 0, 255), 2)
                    cv2.putText(orig_image, pred_classes[j],
                                (int(box[0]), int(box[1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                                2, lineType=cv2.LINE_AA)
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey()


def evaluate_model(model_weights_name, eval_dataset_path, detection_threshold):
    model_compiled = compile_model(model_weights_name)
    result = []
    images_dict = {}
    label_boxes = []
    coordinates = []
    for fn in os.listdir(rf"{eval_dataset_path}\images"):
        im, oim = prepro_img(rf'{eval_dataset_path}\images\{fn}')
        images_dict[im] = oim
    for i, fn in enumerate(os.listdir(rf"{eval_dataset_path}\annotations")):
        tree = ET.parse(rf"{eval_dataset_path}\annotations\{fn}")
        root = tree.getroot()
        object_bndbox = root.find('object').find('bndbox')
        xmin = int(object_bndbox.find('xmin').text)
        ymin = int(object_bndbox.find('ymin').text)
        xmax = int(object_bndbox.find('xmax').text)
        ymax = int(object_bndbox.find('ymax').text)
        label_boxes.append([xmin, ymin, xmax, ymax])
    with torch.no_grad():
        for img in images_dict.keys():
            outputs = model_compiled(img)
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            if len(outputs[0]['boxes']) != 0:
                current_values = []
                for score in outputs[0]['scores']:
                    fl_score = float(str(score)[7:-1])
                    if fl_score > detection_threshold:
                        current_values.append(fl_score)
                if len(current_values) == 1:
                    coordinates.append([t.tolist() for t in outputs[0]['boxes']][0])
                else:
                    result.append(0)
            else:
                result.append(0)
        for i in range(len(coordinates)):
            result.append(cosine_similarity([label_boxes[i]], [coordinates[i]])[0][0])

    prc_file = os.path.join("evaluate", "score.json")
    os.makedirs(os.path.join("evaluate"), exist_ok=True)

    with open(prc_file, "w") as fd:
        json.dump({"Accuracy": sum(result) / len(result)}, fd)

    return f'Accuracy: {sum(result) / len(result)}'


if __name__ == '__main__':
    model = compile_model(r'C:\Users\artyo\PycharmProjects\rcnn\weights_all_classes.pth')
    predict_on_batch(model, r'C:\Users\artyo\PycharmProjects\rcnn\new_eval')
