import torch
from matplotlib import pyplot as plt
import os
import numpy as np
import faiss
from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from PIL import Image, ImageDraw, ImageFont
import cv2


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

COLORS = ["m", "c",  "yellow", "b", "orange", "pink", "plum", "tan", "aqua", "rosybrown"]

# CLASSES = (
#     "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
#     "pottedplant", "sheep", "sofa", "train", "tvmonitor"
# )

CLASSES = (
    "pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle", "traffic light", "traffic sign"
)

def plot_image(ax, img, norm):
    if norm:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
    img = img.astype('uint8')
    ax.imshow(img)


def plot_results(pil_img, prob, boxes, output_dir, classes, targets, ood=False):
    plt.figure(figsize=(16, 10))
    # plt.imshow(pil_img)
    ax = plt.gca()
    image = plot_image(ax, pil_img, True)
    # breakpoint()
    # for p, cl, (xmin, ymin, xmax, ymax), c in zip(prob, classes, boxes.tolist(), COLORS * 100):
    #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
    #                                fill=False, color=c, linewidth=16))
    for p, cl, (xmin, ymin, xmax, ymax) in zip(prob, classes, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=COLORS[cl-1], linewidth=15))
        # cl = p.argmax()
        text = f'{CLASSES[cl-1]}: {p:0.2f}'
        ax.text(xmin, ymin, text, fontsize=17,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    # plt.show()
    print('hhh')

    if ood:
        plt.savefig(os.path.join(output_dir + '/images_ood', f'img_{int(targets[0]["image_id"][0])}.jpg'))
    else:
        plt.savefig(os.path.join(output_dir + '/images', f'img_{int(targets[0]["image_id"][0])}.jpg'))


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox)
    # breakpoint()
    return b

def visualize_prediction_results(samples, result, output_dir, targets, ood):
    # breakpoint()
    probas = result[0]['scores']
    keep = probas > 0.5
    # breakpoint()
    images = samples.tensors[0].cpu().permute(1,2,0).numpy()
    # breakpoint()
    bboxes_scaled = rescale_bboxes(result[0]['original_boxes'], list(images.shape[:2])[::-1])[keep]
    # bboxes_scaled = result[0]['boxes'][keep]

    classes = result[0]['labels'][keep]
    plot_results(images, probas[keep], bboxes_scaled,
                 output_dir, classes, targets, ood)
    # breakpoint()
    return

def visualize_prediction_results_ood(samples, result, output_dir, targets, ood):
    # # breakpoint()
    # probas = result[0]['scores']
    # keep = probas > 0.5
    # # breakpoint()
    # images = samples.tensors[0].cpu().permute(1,2,0).numpy()
    # # breakpoint()
    # bboxes_scaled = rescale_bboxes(result[0]['original_boxes'], list(images.shape[:2])[::-1])[keep]
    # # bboxes_scaled = result[0]['boxes'][keep]
    #
    # classes = result[0]['labels'][keep]
    # plot_results(images, probas[keep], bboxes_scaled, output_dir, classes, targets, ood)
    # # breakpoint()

    # visualize pred
    id_train_data = np.load("/disk1/zl/DDERT_OUTPUTS/bdd_oi_2tasks_ftencoder5e-5_1_0.5/id-pen_maha_train-79.npy")
    length = 256
    print(id_train_data.shape)
    id_train_data = torch.from_numpy(id_train_data).reshape(-1, id_train_data.shape[-1])
    id_train_data = id_train_data[:, :length]

    idx = np.array(range(id_train_data.shape[0]))
    rng = np.random.default_rng(1)
    rng.shuffle(idx)
    index = idx[:10000]
    id_train_data = id_train_data[index]


    boxes_filt, pred_phrases = get_grounding_output(result, CLASSES, 0.1, with_logits=True, train_feat=id_train_data)

    image_pil = load_image(samples.tensors[0].cpu().permute(1,2,0))

    size = image_pil.size
    pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,
    }
    image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]
    image_with_box.save(os.path.join(output_dir + "images_ood", f'img_{int(targets[0]["image_id"][0])}.jpg'))

    return



def load_image(tensor):
    # load image
    numpy = tensor.numpy()
    image_pil = array_to_img(numpy)

    return image_pil


normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
prepos_feat = lambda x: np.ascontiguousarray(np.concatenate([normalizer(x)], axis=1))


def get_grounding_output(prediction, cat_list, box_threshold, with_logits=True, train_feat=None):
    logits = prediction[0]["scores"].cpu()
    # print(prediction[0])
    labels = prediction[0]["labels"]
    boxes = prediction[0]["boxes"].cpu()  # (nq, 4)
    out_pen_features = prediction[0]['pen_features'].squeeze(0).cpu()
    softmax_logits = prediction[0]["logits_for_ood_eval"].squeeze(0).cpu()

    # filter output
    logits_filt = logits.clone()
    softmax_logits_filt = softmax_logits.clone()
    boxes_filt = boxes.clone()
    out_pen_features_filt = out_pen_features.clone()
    filt_mask = logits_filt >= box_threshold
    # nms
    keep = nms(boxes_filt[filt_mask], logits_filt[filt_mask], 0.5)
    logits_filt = logits_filt[filt_mask][keep]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask][keep]  # num_filt, 4
    labels = labels[filt_mask][keep]
    softmax_logits_filt = softmax_logits_filt[filt_mask][keep]
    out_pen_features_filt = out_pen_features_filt[filt_mask][keep].reshape(-1, 256)

    # get phrase
    # build pred
    pred_phrases = []
    assert train_feat is not None
    # preprocess id training features
    id_train_data = prepos_feat(train_feat)
    index = faiss.IndexFlatL2(id_train_data.shape[1])
    index.add(id_train_data)
    index.add(id_train_data)
    for idx, (logit, box, feat, softmax_logit) in enumerate(zip(logits_filt, boxes_filt, out_pen_features_filt, softmax_logits_filt)):
        # energy = -torch.logsumexp(logit, dim=-1)
        # if energy > 1:
        #     pred_phrases.append('OoD' + f"({str(logit.max().item())[:4]})")
        feat = feat.reshape(-1, 256)
        feat = prepos_feat(feat)
        D, _ = index.search(feat, 1)
        scores_in = -D[:, -1]
        # print(scores_in)
        if scores_in < -0.144: # -0.58 for baseline OI  -0.62 for our OI 
            pred_phrases.append('OoD')
        else:
            pred_phrase = cat_list[int(labels[idx])-1]
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(softmax_logit.softmax(-1).max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    return boxes_filt, pred_phrases


##################################################################
from torch import Tensor
import torch


def box_area(boxes: Tensor) -> Tensor:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Arguments:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format

    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    area1 = box_area(boxes1)  # N
    area2 = box_area(boxes2)  # M

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou  # NxM


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float):
    """
    :param boxes: [N, 4]
    :param scores: [N]
    :param iou_threshold: 0.7
    :return:
    """
    keep = []
    idxs = scores.argsort()

    while idxs.numel() > 0:
        max_score_index = idxs[-1].reshape(-1)
        # print(idxs, max_score_index, boxes.shape)
        max_score_box = boxes[max_score_index]  # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]  # [?, 4]
        # print(max_score_box.shape)
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]

    keep = idxs.new(keep)  # Tensor
    # source_tensor = torch.zeros_like(scores)
    # source_tensor.index_fill_(0, keep, 1)
    return keep
############################################################################




def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # random color
        if "OoD" in str(label):
            color = tuple([0, 255, 0])  # green
        else:
            color = tuple([255, 0, 0])  # red
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=20)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        # font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
        # font = ImageFont.truetype(font_path, size=600)
        
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        font = ImageFont.truetype("/home/zl/arial.ttf", 70)
        draw.text((x0, y0), str(label), font=font, fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=20)

    return image_pil, mask


def array_to_img(x, scale=True):
    """
    Converts a 3D Numpy array to a PIL Image instance.
    """

    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])



