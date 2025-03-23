# tensorflow_unet3d.py
import tensorflow as tf
from keras import layers, models


def build_tf_unet3d(input_shape, use_physics=False, init_features=64):
    """
    Build a 3D U-Net model using TensorFlow/Keras.
    
    Args:
        input_shape: Tuple of (depth, height, width, channels)
        use_physics: Whether to use physics-informed neural network
        init_features: Number of initial features in the first layer
        
    Returns:
        TensorFlow model
    """
    # Print debug info
    print(f"Building TensorFlow U-Net with input shape: {input_shape}")
    
    # Ensure input shape has 4 dimensions
    if len(input_shape) != 4:
        raise ValueError(f"Expected 4D input shape (D,H,W,C), got {input_shape}")
    
    # Create model input
    inputs = layers.Input(shape=input_shape)
    
    # Encoder path (downsampling)
    # Level 1
    conv1 = layers.Conv3D(init_features, 3, padding='same', activation='relu')(inputs)
    conv1 = layers.Conv3D(init_features, 3, padding='same', activation='relu')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    
    # Level 2
    conv2 = layers.Conv3D(init_features*2, 3, padding='same', activation='relu')(pool1)
    conv2 = layers.Conv3D(init_features*2, 3, padding='same', activation='relu')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    
    # Level 3
    conv3 = layers.Conv3D(init_features*4, 3, padding='same', activation='relu')(pool2)
    conv3 = layers.Conv3D(init_features*4, 3, padding='same', activation='relu')(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    
    # Level 4
    conv4 = layers.Conv3D(init_features*8, 3, padding='same', activation='relu')(pool3)
    conv4 = layers.Conv3D(init_features*8, 3, padding='same', activation='relu')(conv4)
    pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)
    
    # Bottleneck
    bottleneck = layers.Conv3D(init_features*16, 3, padding='same', activation='relu')(pool4)
    bottleneck = layers.Conv3D(init_features*16, 3, padding='same', activation='relu')(bottleneck)
    
    # Decoder path (upsampling)
    # Level 4
    up4 = layers.UpSampling3D(size=(2, 2, 2))(bottleneck)
    up4 = layers.Conv3D(init_features*8, 2, padding='same', activation='relu')(up4)
    concat4 = layers.Concatenate(axis=-1)([up4, conv4])
    conv_up4 = layers.Conv3D(init_features*8, 3, padding='same', activation='relu')(concat4)
    conv_up4 = layers.Conv3D(init_features*8, 3, padding='same', activation='relu')(conv_up4)
    
    # Level 3
    up3 = layers.UpSampling3D(size=(2, 2, 2))(conv_up4)
    up3 = layers.Conv3D(init_features*4, 2, padding='same', activation='relu')(up3)
    concat3 = layers.Concatenate(axis=-1)([up3, conv3])
    conv_up3 = layers.Conv3D(init_features*4, 3, padding='same', activation='relu')(concat3)
    conv_up3 = layers.Conv3D(init_features*4, 3, padding='same', activation='relu')(conv_up3)
    
    # Level 2
    up2 = layers.UpSampling3D(size=(2, 2, 2))(conv_up3)
    up2 = layers.Conv3D(init_features*2, 2, padding='same', activation='relu')(up2)
    concat2 = layers.Concatenate(axis=-1)([up2, conv2])
    conv_up2 = layers.Conv3D(init_features*2, 3, padding='same', activation='relu')(concat2)
    conv_up2 = layers.Conv3D(init_features*2, 3, padding='same', activation='relu')(conv_up2)
    
    # Level 1
    up1 = layers.UpSampling3D(size=(2, 2, 2))(conv_up2)
    up1 = layers.Conv3D(init_features, 2, padding='same', activation='relu')(up1)
    concat1 = layers.Concatenate(axis=-1)([up1, conv1])
    conv_up1 = layers.Conv3D(init_features, 3, padding='same', activation='relu')(concat1)
    conv_up1 = layers.Conv3D(init_features, 3, padding='same', activation='relu')(conv_up1)
    
    # Output layer
    outputs = layers.Conv3D(1, 1, padding='same', activation='sigmoid')(conv_up1)
    
    # Add physics-informed components if requested
    if use_physics:
        # Simple physics branch from the output of the last layer
        physics_branch = layers.Conv3D(init_features, 3, padding='same', activation='relu')(conv_up1)
        physics_branch = layers.Conv3D(init_features, 3, padding='same', activation='relu')(physics_branch)
        physics_output = layers.Conv3D(1, 1, padding='same', activation='sigmoid')(physics_branch)
        
        # Combine main output with physics-informed output
        outputs = layers.Add()([outputs, physics_output])
    
    # Create model
    model = models.Model(inputs, outputs)
    
    return model

def tf_compute_n_ssim(prediction, ground_truth, input_image):
    """
    Compute Normalized SSIM using TensorFlow
    
    Args:
        prediction: Predicted 3D volume
        ground_truth: Ground truth 3D volume
        input_image: Input 3D volume (single view)
    
    Returns:
        float: Normalized SSIM score
    """
    # Convert numpy arrays to TensorFlow tensors if needed
    if not isinstance(prediction, tf.Tensor):
        prediction = tf.convert_to_tensor(prediction, dtype=tf.float32)
    if not isinstance(ground_truth, tf.Tensor):
        ground_truth = tf.convert_to_tensor(ground_truth, dtype=tf.float32)
    if not isinstance(input_image, tf.Tensor):
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
    
    # Ensure tensors are float32
    prediction = tf.cast(prediction, tf.float32)
    ground_truth = tf.cast(ground_truth, tf.float32)
    input_image = tf.cast(input_image, tf.float32)
    
    # Compute SSIM slice by slice (TensorFlow's SSIM works on 2D images)
    ssim_values_pred = []
    ssim_values_input = []
    
    # Loop through each z-slice
    for z in range(prediction.shape[0]):
        # Extract slices
        pred_slice = prediction[z:z+1, :, :]  # Keep dimensions for SSIM
        gt_slice = ground_truth[z:z+1, :, :]
        input_slice = input_image[z:z+1, :, :]
        
        # Compute SSIM
        ssim_pred = tf.image.ssim(pred_slice, gt_slice, max_val=1.0)
        ssim_input = tf.image.ssim(input_slice, gt_slice, max_val=1.0)
        
        ssim_values_pred.append(ssim_pred)
        ssim_values_input.append(ssim_input)
    
    # Average SSIM values
    prediction_ssim = tf.reduce_mean(ssim_values_pred)
    reference_ssim = tf.reduce_mean(ssim_values_input)
    
    # Compute normalized SSIM
    n_ssim = (prediction_ssim - reference_ssim) / (1.0 - reference_ssim)
    
    return n_ssim.numpy()