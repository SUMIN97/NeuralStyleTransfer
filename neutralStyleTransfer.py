import tensorflow as tf
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

# content_image = tf.constant(load_sample_image('china.jpg'))
drawing_image = tf.constant(load_sample_image('Drawing/0.jpg'))
oilpaint_image = tf.constant(load_sample_image('OilPaint/0.jpg'))


def preprocess_img(img):
    "이미지 가로 길이를 512로 통일"

    # 0~255 uint 자료형을 0 ~ 1 실수 자로형으로 변환한다.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 이미지 크기
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    # 이미지의 가로, 세로 중 긴 변의 값(가로 길이)
    long_dim = max(shape)
    # 이미지의 목표 가로 길이
    max_dim = 512
    # 목표 크기와의 비율
    scale = max_dim / long_dim
    # 새 크기
    new_shape = tf.cast(shape * scale, tf.int32)
    # 이미지 크기 변화
    img = tf.image.resize(img, new_shape)
    # 1장짜리 배치 데이터(4차원 텐서)로 변환
    img = img[tf.newaxis, :]

    return img


drawing_image = preprocess_img(drawing_image)
oilpaint_image = preprocess_img(oilpaint_image)

plt.subplot(121)
plt.imshow(tf.squeeze(drawing_image))
plt.title("소묘 이미지")
plt.subplot(122)
plt.imshow(tf.squeeze(oilpaint_image))
plt.title("유 이미지")
# plt.show()

#스타일 전이 구현
from tensorflow.keras.applications import VGG19

vgg = VGG19(include_top = False, weights= 'imagenet')
vgg.trainable = False

#Content layer where will pull our feature maps
content_layers = ['block5_conv2']

#Style layer of interest
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

from tensorflow.keras.applications.vgg19 import preprocess_input

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result/(num_locations)

class StyleModel(tf.keras.models.Model):
    def __init__(self, style_layers):
        "VGG 모형 입력으로 부터 스타일 레이어와 내용 레이어를 출력하는 모형을 만든다."
        super(StyleModel, self).__init__()
        # self.content_layers = content_layers화
        self.style_layers = style_layers
        self.num_style_layers = len(style_layers)
        outputs = [vgg.get_layer(name).output for name in style_layers ]
        self.vgg = tf.keras.Model([vgg.input], outputs)
        self.vgg.trainable = False

        def call(self, inputs):
            "입력"
            #이미지의 0~1 입력을 0~255로 확대
            inputs = inputs *255.0
            #데이터의 평균으로 정규
            preprocessed_input = preprocess_input(inputs)
            #스타일 레이어와 컨텐트 레이어의 출력
            style_outputs = self.vgg(preprocessed_input)

            #그램 행렬로 각 레이어의 스타일 출력값 계산
            style_outputs = [gram_matrix(style_outputs) for style_output in style_outputs]

            #content, style dict
            style_dict = {style_name:value for style_name, value in zip(self.num_style_layers, style_outputs)}

            return {'style':style_dict}

extractor = StyleModel(style_layers)
#참조 이미지의 스타일 레이어
drawing_targets = extractor(drawing_image)['style']
#목표 이미지의 내용 레이어
oilpaint_targets = extractor(oilpaint_image)['style']

style_weight = 1e-2
content_weight = 1e4

def style_loss(outputs):
    #outputs는 현재 이미지의 스타일 및 내용 출력
    drawing_outputs = outputs['drawing']
    oilpaint_outputs = outputs['oilpaint']

    drawing_loss = tf.add_n([tf.reduce_mean((drawing_outputs[name] - drawing_targets[name]) **2)
                           for name in drawing_outputs.keys()])

    drawing_loss *=style_weight/ num_style_layers

    #현재 이미지의 내용 출력과 목표 이미지의 내용 출력 차이 계산
    oilpaint_loss = tf.add_n([tf.reduce_mean((oilpaint_outputs[name] - oilpaint_targets[name]) ** 2)
                           for name in oilpaint_outputs.keys()])
    oilpaint_loss *= content_weight/ num_content_layers

    loss = {'drawing':drawing_loss, 'oilpaint':oilpaint_loss}
    return loss

opt = tf.optimizers.Adam(learning_rate= 0.02, beta_1 = 0.99, epsilon=1e-1)

def train_step(content_drawing_image, content_oilpaint_image):
    with tf.GradientTape() as tape:
            #현재 이미지 스타일 레이어 및 내용 레이어 출력을 계산
            drawing_outputs = extractor(content_drawing_image)
            oilpaint_outputs = extractor(content_oilpaint_image)
            outputs = {'drawing':drawing_outputs, 'oilpaint':oilpaint_outputs}
            loss = style_loss(outputs)

            grad = tape.gradient(loss['drawing'], content_drawing_image)
            opt.apply_gradients([(grad, content_drawing_image)])

            grad = tape.gradient(loss['oilpaint'], content_oilpaint_image)
            opt.apply_gradients([(grad, content_oilpaint_image)])

            #새로운 이미지 저장
            image_clipped = tf.clip_by_value(image, clip_value_min= 0.0, clip_value_max= 1.0)
            image.assign(image_clipped)

content_drawing_image = tf.Variable(drawing_image)
content_oilpaint_image = tf.Varialbe(oilpaint_image)
for n in range(200):
    train_step(content_drawing_image, content_oilpaint_image)


plt.imshow(tf.squeeze(image))
plt.show()




