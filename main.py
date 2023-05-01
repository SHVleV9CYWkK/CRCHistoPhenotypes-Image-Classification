import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('./Image_classification_data/data_labels_mainData.csv')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data['cellType'] = train_data['cellType'].astype(str)
test_data['cellType'] = test_data['cellType'].astype(str)
# 加载图像数据
def load_images(image_paths, target_size=(32, 32)):
    images = []
    for path in image_paths:
        img = load_img(path, target_size=target_size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

# 定义Transformer的基本组件
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

    def call(self, inputs):
        return self.attention(inputs, inputs)

def TransformerBlock(embed_dim, num_heads, ff_dim):
    inputs = layers.Input(shape=(None, embed_dim))
    attn = MultiHeadSelfAttention(embed_dim, num_heads)(inputs)
    attn = layers.LayerNormalization(epsilon=1e-6)(attn)
    outputs = layers.Add()([inputs, attn])
    ffn = layers.Dense(ff_dim, activation="relu")(outputs)
    ffn = layers.Dense(embed_dim)(ffn)
    ffn = layers.LayerNormalization(epsilon=1e-6)(ffn)
    outputs = layers.Add()([outputs, ffn])

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# 定义图像分类模型
def create_vision_transformer():
    inputs = layers.Input(shape=(27, 27, 3))
    # 将图像reshape为序列
    x = layers.Reshape((27 * 27, 3))(inputs)
    # 嵌入层
    x = layers.Dense(128, activation="relu")(x)
    # Transformer层
    x = TransformerBlock(128, 4, 512)(x)
    # 平均池化
    x = layers.GlobalAveragePooling1D()(x)
    # 输出层
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)


num_classes = 4  # 将其更改为您数据集中的类别数量
batch_size = 32

# 数据预处理
# x_train = load_images(train_image_paths) / 255.0
# x_val = load_images(val_image_paths) / 255.0
# y_train = to_categorical(train_labels, num_classes)
# y_val = to_categorical(val_labels, num_classes)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    directory='./Image_classification_data/patch_images',
    x_col='ImageName',
    y_col='cellType',
    target_size=(27, 27),
    batch_size=batch_size,
    class_mode='categorical')
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

validation_generator = test_datagen.flow_from_dataframe(
    test_data,
    directory='./Image_classification_data/patch_images',
    x_col='ImageName',
    y_col='cellType',
    target_size=(27, 27),
    batch_size=batch_size,
    class_mode='categorical')


# 训练和评估模型
model = create_vision_transformer()
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_generator, batch_size=64, epochs=50, validation_data=validation_generator)
