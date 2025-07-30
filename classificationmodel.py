import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 1. 경로 설정
train_dir = 'train'
val_dir = 'valid'

# 2. 데이터 전처리 및 증강 설정
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True)

val_data = val_gen.flow_from_directory(
    val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# 3. Functional API 모델 정의
def create_model(num_classes):
    inputs = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = True

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = create_model(train_data.num_classes)

# 4. 컴파일
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 콜백 설정
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_trash_classifier.keras", save_best_only=True)
]

# 6. 학습
model.fit(train_data, validation_data=val_data, epochs=40, callbacks=callbacks, verbose=1)
model.save("final_trash_classifier.keras")


# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras import layers, models
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# import numpy as np

# # 1. 경로 설정
# train_dir = 'train'
# val_dir = 'valid'

# # 2. 데이터 전처리 및 증강 설정
# train_gen = ImageDataGenerator(
#     rescale=1./255,  # 정규화
#     rotation_range=30,
#     zoom_range=0.3,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.8, 1.2],
#     fill_mode='nearest'
# )
# val_gen = ImageDataGenerator(rescale=1./255)

# # 3. 이미지 불러오기
# train_data = train_gen.flow_from_directory(
#     train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=True)

# val_data = val_gen.flow_from_directory(
#     val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# # 4. 사전학습된 MobileNetV2 기반 전이학습 (Transfer Learning)
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = True  # 전체 레이어 학습 허용

# # 5. 모델 구성
# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.BatchNormalization(),
#     layers.Dropout(0.2),
#     layers.Dense(256, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(train_data.num_classes, activation='softmax')  # 클래스 수만큼 출력
# ])

# # 6. 모델 컴파일
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# # 7. 콜백 설정: 조기 종료 + 체크포인트 저장
# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
#     ModelCheckpoint("best_trash_classifier.keras", save_best_only=True)
# ]

# # 8. 학습 시작
# history = model.fit(
#     train_data,
#     validation_data=val_data,
#     epochs=40,
#     callbacks=callbacks,
#     verbose=1  # 출력 방식 설정 (0, 1, 2 가능)
# )

# # 9. 모델 저장
# model.save("final_trash_classifier.keras")

# 10. 성능 평가
val_data.reset()
pred_probs = model.predict(val_data)
pred_classes = np.argmax(pred_probs, axis=1)
true_classes = val_data.classes
class_labels = list(val_data.class_indices.keys())

# 11. 혼동 행렬 출력
cm = confusion_matrix(true_classes, pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
disp.plot(xticks_rotation=45, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 12. 분류 리포트 출력
print("\nClassification Report:")
print(classification_report(true_classes, pred_classes, target_names=class_labels))


# # 학습 후 자동 변환 
# model.save("final_trash_classifier.keras")