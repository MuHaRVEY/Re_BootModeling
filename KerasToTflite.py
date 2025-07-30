import tensorflow as tf
import keras

# 1. 모델 불러오기
model = keras.models.load_model("final_trash_classifier.keras")

# 2. 변환기 설정
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 3. 변환 옵션 (에러 회피용)
converter._experimental_lower_tensor_list_ops = False  # 이 옵션이 없는 경우 생략

# 4. 변환 실행
tflite_model = converter.convert()

# 5. 저장
with open("trash_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("TFLite 변환 성공!")
