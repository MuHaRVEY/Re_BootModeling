import tensorflow as tf

# 저장된 Keras 모델 로드
model = tf.keras.models.load_model("final_trash_classifier.keras")

# TFLite 변환기 생성
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 최적화 (선택사항, 용량 줄이기)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# TFLite 모델로 변환
tflite_model = converter.convert()

# .tflite 파일로 저장
with open("trash_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite model saved: trash_classifier.tflite")
