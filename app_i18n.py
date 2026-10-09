LANG_LABELS = {
    "vi": "Tiếng Việt",
    "zh": "中文",
    "en": "English",
}

TEXT = {
    "vi": {
        "subtitle": "Gán một nhãn cho cả tấm ảnh. Không khoanh từng vật trong hình.",
        "language": "Ngôn ngữ",
        "checkpoint": "Checkpoint",
        "checkpoint_help": "File .pth sau khi train. Tên class lấy từ folder dataset.",
        "upload": "Chọn ảnh",
        "empty": "Cần models/best_model.pth, rồi tải một ảnh lên.",
        "result": "Kết quả",
        "confidence": "Độ tin cậy",
        "image": "Ảnh",
        "schema": "Sơ đồ minh họa",
        "schema_note": "INPUT → CONV → FEATURES → CLASSES. Nốt lớp ra theo Softmax, không phải neuron thật trong CNN.",
        "settings": "Cài đặt",
        "stage_empty": "Thả ảnh vào khung dưới",
    },
    "zh": {
        "subtitle": "给整张图片一个类别。不会框出图中的每个物体。",
        "language": "语言",
        "checkpoint": "Checkpoint",
        "checkpoint_help": "训练后的 .pth 文件。类别名来自数据集文件夹。",
        "upload": "选择图片",
        "empty": "需要 models/best_model.pth，然后上传一张图片。",
        "result": "结果",
        "confidence": "置信度",
        "image": "图片",
        "schema": "示意图",
        "schema_note": "INPUT → CONV → FEATURES → CLASSES。输出节点按 Softmax，不是 CNN 内部真实神经元。",
        "settings": "设置",
        "stage_empty": "把图片放到下面的画框",
    },
    "en": {
        "subtitle": "Assigns one label to the whole image. It does not box objects in the photo.",
        "language": "Language",
        "checkpoint": "Checkpoint",
        "checkpoint_help": "The .pth file from training. Class names come from dataset folders.",
        "upload": "Choose an image",
        "empty": "You need models/best_model.pth, then upload one image.",
        "result": "Result",
        "confidence": "Confidence",
        "image": "Image",
        "schema": "Schematic",
        "schema_note": "INPUT → CONV → FEATURES → CLASSES. Output nodes follow Softmax, not real neurons inside the CNN.",
        "settings": "Settings",
        "stage_empty": "Drop an image into the frame below",
    },
}


def translate(lang: str, key: str) -> str:
    if lang not in TEXT:
        lang = "vi"
    if key not in TEXT[lang]:
        raise KeyError(f"Missing text key {key!r} for {lang}")
    return TEXT[lang][key]
