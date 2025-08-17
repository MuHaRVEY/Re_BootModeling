# fast_trash_train_and_export.py
import os, json, math
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.data import AUTOTUNE
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

# --- 파라미터 조절
IMG_SIZE = 192          # 224 -> 192로 내려 속도↑(원하면 160/224 조절)
ALPHA    = 0.75         # MobileNetV2 폭: 1.0(기본)보다 0.75/0.5가 더 가벼움
BATCH    = 32
EPOCHS_WARMUP = 3       # 베이스 동결 워밍업
EPOCHS_FINETUNE = 20    # 파인튜닝
LEARN_WARMUP = 3e-4
LEARN_FINETUNE = 2e-4
MIXED_PRECISION = False # GPU면 True 권장. CPU는 보통 이득 적음.
# =========================================

from pathlib import Path

DATA_ROOT = Path(r"C:\Users\rkddn\trash.v1i.folder")   # <- 여기에 옮긴 경로
train_dir = str((DATA_ROOT / "train").resolve())
valid_dir = str((DATA_ROOT / "valid").resolve())
test_dir  = str((DATA_ROOT / "test").resolve())  # 없으면 무시됨

# (선택) 빠른 체크
for p in [train_dir, valid_dir]:
    if not Path(p).exists():
        raise FileNotFoundError(f"경로 없음: {p}")


if MIXED_PRECISION:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# -------- tf.data로 로딩 (빠름) --------
def make_ds(root, shuffle, batch, img_size):
    ds = tf.keras.preprocessing.image_dataset_from_directory(
        root,
        image_size=(img_size, img_size),
        label_mode="categorical",
        batch_size=batch,
        shuffle=shuffle
    )
    return ds

train_ds = make_ds(train_dir, True,  BATCH, IMG_SIZE)
val_ds   = make_ds(valid_dir, False, BATCH, IMG_SIZE)
num_classes = train_ds.cardinality().numpy()  # cardinality는 배치 수
class_names = train_ds.class_names
N_CLASSES = len(class_names)

# 증강- 그래프 내에서
data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1)
], name="augment")

# 전처리 파이프라인 (cache→shuffle→prefetch)
def prep_train(x, y):
    x = tf.cast(x, tf.float32) / 255.
    x = data_augment(x, training=True)
    return x, y

def prep_eval(x, y):
    x = tf.cast(x, tf.float32) / 255.
    return x, y

train_ds = (train_ds
            .map(prep_train, num_parallel_calls=AUTOTUNE)
            .cache()
            .prefetch(AUTOTUNE))

val_ds = (val_ds
          .map(prep_eval, num_parallel_calls=AUTOTUNE)
          .cache()
          .prefetch(AUTOTUNE))

# 여기부터 모델
def build_model(n_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base = MobileNetV2(
        input_tensor=inputs,
        include_top=False,
        weights="imagenet",
        alpha=ALPHA
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax", dtype="float32")(x)  # mixed일 때 출력은 float32로

    model = models.Model(inputs, outputs)
    return model

model = build_model(N_CLASSES)

# -------- 콜백 ----------
ckpt_path = "best_trash_classifier.keras"
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
]

# -------- 1) 워밍업(베이스 동결) --------
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name:
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARN_WARMUP),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_WARMUP, callbacks=callbacks, verbose=1)

# -------- 2) 파인튜닝(상위 블록만 풀기) --------
# MobileNetV2의 끝쪽 블록 몇 개만 학습(속도와 안정성 균형)
for layer in model.layers:
    if "block_" in layer.name:
        # block_12 ~ block_16 정도만 학습
        try:
            block_idx = int(layer.name.split("block_")[1].split("_")[0])
            layer.trainable = (block_idx >= 12)
        except:
            pass

model.compile(
    optimizer=tf.keras.optimizers.Adam(LEARN_FINETUNE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINETUNE, callbacks=callbacks, verbose=1)

# 최종 저장
model.save("final_trash_classifier.keras")

# -------- 검증셋 평가/리포트(빠른 확인) --------
val_np = list(val_ds.unbatch().as_numpy_iterator())
xv = np.stack([a for a, _ in val_np])
yv = np.stack([b for _, b in val_np])
pred = model.predict(val_ds, verbose=0)
pred_cls = pred.argmax(1)
true_cls = yv.argmax(1)

cm = confusion_matrix(true_cls, pred_cls)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title("Confusion Matrix (VALID)")
plt.show()

print("\nClassification Report (VALID):")
print(classification_report(true_cls, pred_cls, target_names=class_names))

# -------- (선택) TEST셋 평가 --------
if os.path.isdir(test_dir):
    test_ds = make_ds(test_dir, False, BATCH, IMG_SIZE)
    test_ds = test_ds.map(prep_eval, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    print("\nTest evaluation:")
    model.evaluate(test_ds, verbose=1)

# -------- TFLite 변환 (빠르고 안전한 from_keras_model) --------
def export_tflite(m, fname="trash_classifier.tflite", quant="dynamic"):
    converter = tf.lite.TFLiteConverter.from_keras_model(m)
    if quant == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(fname, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite saved → {fname}")

# 기본: 동적 양자화(대부분 CPU에서 속도/크기 이득)
export_tflite(model, "trash_classifier_dynamic.tflite", quant="dynamic")
# 용량 더 줄이고 싶으면(모바일 GPU/NNAPI에서 빠른 편)
export_tflite(model, "trash_classifier_fp16.tflite", quant="float16")

# 라벨 저장(모바일에서 사용)
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print("labels.json saved.")
