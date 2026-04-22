import tensorflow as tf

def augment(image, label, img_size=(299, 299)):
    """Apply random augmentations to a single image.
    
    No horizontal flip: flipping would change hand orientation
    and alter the meaning of ASL signs.
    """
    # Random brightness adjustment (+/- 20%)
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Random contrast adjustment (0.8x to 1.2x)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random saturation adjustment
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # Random zoom via crop-and-resize (0.8x to 1.0x)
    crop_factor = tf.random.uniform([], minval=0.8, maxval=1.0)
    orig_height = tf.shape(image)[0]
    orig_width = tf.shape(image)[1]
    crop_h = tf.cast(tf.cast(orig_height, tf.float32) * crop_factor, tf.int32)
    crop_w = tf.cast(tf.cast(orig_width, tf.float32) * crop_factor, tf.int32)

    image = tf.image.random_crop(image, size=[crop_h, crop_w, 3])
    image = tf.image.resize(image, img_size)

    # Random translation via pad-and-crop (+/- 10%)
    pad_amount = int(0.1 * img_size[0])
    image = tf.image.resize_with_crop_or_pad(
        image,
        img_size[0] + pad_amount,
        img_size[1] + pad_amount
    )
    image = tf.image.random_crop(image, size=[img_size[0], img_size[1], 3])

    # Clip to valid range
    image = tf.clip_by_value(image, 0.0, 255.0)

    return image, label
