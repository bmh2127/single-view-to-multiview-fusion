def analyze_gradient_direction(volume):
    """
    Analyze and detect the gradient direction in a 3D volume.
    
    This function analyzes a 3D volume and detects the primary direction of
    intensity change (gradient) in Z, X, and Y directions. This information
    can be used to normalize or standardize the gradient direction.
    
    Args:
        volume: 3D numpy array of shape (Z, Y, X)
        
    Returns:
        dict: Dictionary containing gradient direction information:
             - 'z_direction': Direction of gradient in Z ('increasing', 'decreasing', or 'none')
             - 'x_direction': Direction of gradient in X ('increasing', 'decreasing', or 'none')
             - 'y_direction': Direction of gradient in Y ('increasing', 'decreasing', or 'none')
             - 'primary_axis': The axis with the strongest gradient ('z', 'y', 'x', or 'none')
             - 'z_gradient_strength': Relative strength of Z gradient
             - 'y_gradient_strength': Relative strength of Y gradient
             - 'x_gradient_strength': Relative strength of X gradient
    """
    import numpy as np
    import logging
    
    # Ensure we have a 3D volume
    if len(volume.shape) < 3:
        logging.warning(f"Expected 3D volume, got shape {volume.shape}. Adding Z dimension.")
        volume = volume[np.newaxis, ...]
    elif len(volume.shape) > 3:
        logging.warning(f"Expected 3D volume, got shape {volume.shape}. Using first channel.")
        volume = volume[..., 0]  # Assuming 4th dimension is channel
    
    # Get volume dimensions
    z_dim, y_dim, x_dim = volume.shape
    
    # Calculate mean intensity along each axis
    z_profile = np.mean(volume, axis=(1, 2))  # Average across Y and X for each Z
    y_profile = np.mean(volume, axis=(0, 2))  # Average across Z and X for each Y
    x_profile = np.mean(volume, axis=(0, 1))  # Average across Z and Y for each X
    
    # Calculate gradient direction and strength for Z axis
    if z_dim > 1:
        z_gradient = np.gradient(z_profile)
        z_gradient_mean = np.mean(z_gradient)
        z_gradient_strength = np.abs(z_gradient_mean)
        
        if z_gradient_strength < 0.01 * np.mean(z_profile):  # Threshold to detect flat gradients
            z_direction = 'none'
        elif z_gradient_mean > 0:
            z_direction = 'increasing'
        else:
            z_direction = 'decreasing'
    else:
        z_direction = 'none'
        z_gradient_strength = 0.0
    
    # Calculate gradient direction and strength for Y axis
    if y_dim > 1:
        y_gradient = np.gradient(y_profile)
        y_gradient_mean = np.mean(y_gradient)
        y_gradient_strength = np.abs(y_gradient_mean)
        
        if y_gradient_strength < 0.01 * np.mean(y_profile):
            y_direction = 'none'
        elif y_gradient_mean > 0:
            y_direction = 'increasing'
        else:
            y_direction = 'decreasing'
    else:
        y_direction = 'none'
        y_gradient_strength = 0.0
    
    # Calculate gradient direction and strength for X axis
    if x_dim > 1:
        x_gradient = np.gradient(x_profile)
        x_gradient_mean = np.mean(x_gradient)
        x_gradient_strength = np.abs(x_gradient_mean)
        
        if x_gradient_strength < 0.01 * np.mean(x_profile):
            x_direction = 'none'
        elif x_gradient_mean > 0:
            x_direction = 'increasing'
        else:
            x_direction = 'decreasing'
    else:
        x_direction = 'none'
        x_gradient_strength = 0.0
    
    # Determine the primary axis with the strongest gradient
    strengths = {
        'z': z_gradient_strength,
        'y': y_gradient_strength,
        'x': x_gradient_strength
    }
    
    primary_axis = max(strengths, key=strengths.get)
    if strengths[primary_axis] < 0.01 * np.mean(volume):  # Overall threshold
        primary_axis = 'none'
    
    # Normalize gradient strengths to sum to 1
    total_strength = z_gradient_strength + y_gradient_strength + x_gradient_strength
    if total_strength > 0:
        z_gradient_strength_normalized = z_gradient_strength / total_strength
        y_gradient_strength_normalized = y_gradient_strength / total_strength
        x_gradient_strength_normalized = x_gradient_strength / total_strength
    else:
        z_gradient_strength_normalized = 0.0
        y_gradient_strength_normalized = 0.0
        x_gradient_strength_normalized = 0.0
    
    # Return the analysis results
    return {
        'z_direction': z_direction,
        'y_direction': y_direction,
        'x_direction': x_direction,
        'primary_axis': primary_axis,
        'z_gradient_strength': z_gradient_strength_normalized,
        'y_gradient_strength': y_gradient_strength_normalized,
        'x_gradient_strength': x_gradient_strength_normalized
    }


def normalize_gradient_direction(volume, target_directions=None):
    """
    Normalize the gradient direction in a 3D volume to a standard orientation.
    
    This function detects the gradient direction in each axis and flips the volume
    to match the specified target directions.
    
    Args:
        volume: 3D numpy array of shape (Z, Y, X)
        target_directions: Dictionary specifying desired gradient directions
                          e.g., {'z_direction': 'increasing', 'y_direction': 'none', 'x_direction': 'none'}
                          Default is {'z_direction': 'increasing', 'y_direction': 'increasing', 'x_direction': 'increasing'}
    
    Returns:
        tuple: (normalized_volume, gradient_info)
               - normalized_volume: Volume with standardized gradient directions
               - gradient_info: Dictionary with gradient analysis information
    """
    import numpy as np
    
    # Default target directions (standardize to increasing gradients)
    if target_directions is None:
        target_directions = {
            'z_direction': 'increasing',
            'y_direction': 'increasing',
            'x_direction': 'increasing'
        }
    
    # Analyze current gradient directions
    gradient_info = analyze_gradient_direction(volume)
    
    # Create a copy of the volume to avoid modifying the original
    normalized_volume = volume.copy()
    
    # Flip Z axis if needed
    if gradient_info['z_direction'] != 'none' and target_directions['z_direction'] != 'none':
        if gradient_info['z_direction'] != target_directions['z_direction']:
            normalized_volume = normalized_volume[::-1, :, :]
            gradient_info['z_direction'] = target_directions['z_direction']
    
    # Flip Y axis if needed
    if gradient_info['y_direction'] != 'none' and target_directions['y_direction'] != 'none':
        if gradient_info['y_direction'] != target_directions['y_direction']:
            normalized_volume = normalized_volume[:, ::-1, :]
            gradient_info['y_direction'] = target_directions['y_direction']
    
    # Flip X axis if needed
    if gradient_info['x_direction'] != 'none' and target_directions['x_direction'] != 'none':
        if gradient_info['x_direction'] != target_directions['x_direction']:
            normalized_volume = normalized_volume[:, :, ::-1]
            gradient_info['x_direction'] = target_directions['x_direction']
    
    return normalized_volume, gradient_info


# Example usage and visualization code
def visualize_gradient_analysis(volume, gradient_info=None):
    """
    Visualize the gradient analysis by plotting intensity profiles along each axis.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        gradient_info: Optional dictionary from analyze_gradient_direction
    
    Returns:
        matplotlib.figure.Figure: Figure with the visualizations
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Ensure we have a 3D volume
    if len(volume.shape) < 3:
        volume = volume[np.newaxis, ...]
    elif len(volume.shape) > 3:
        volume = volume[..., 0]
    
    # Calculate mean intensity along each axis
    z_profile = np.mean(volume, axis=(1, 2))
    y_profile = np.mean(volume, axis=(0, 2))
    x_profile = np.mean(volume, axis=(0, 1))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Z profile
    axes[0].plot(z_profile)
    axes[0].set_title(f"Z Profile ({volume.shape[0]} slices)")
    axes[0].set_xlabel("Z position")
    axes[0].set_ylabel("Mean intensity")
    
    # Plot Y profile
    axes[1].plot(y_profile)
    axes[1].set_title(f"Y Profile ({volume.shape[1]} pixels)")
    axes[1].set_xlabel("Y position")
    axes[1].set_ylabel("Mean intensity")
    
    # Plot X profile
    axes[2].plot(x_profile)
    axes[2].set_title(f"X Profile ({volume.shape[2]} pixels)")
    axes[2].set_xlabel("X position")
    axes[2].set_ylabel("Mean intensity")
    
    # Add gradient direction info as text if provided
    if gradient_info:
        axes[0].text(0.05, 0.85, f"Direction: {gradient_info['z_direction']}\nStrength: {gradient_info['z_gradient_strength']:.2f}",
                    transform=axes[0].transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        axes[1].text(0.05, 0.85, f"Direction: {gradient_info['y_direction']}\nStrength: {gradient_info['y_gradient_strength']:.2f}",
                    transform=axes[1].transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        axes[2].text(0.05, 0.85, f"Direction: {gradient_info['x_direction']}\nStrength: {gradient_info['x_gradient_strength']:.2f}",
                    transform=axes[2].transAxes, bbox=dict(facecolor='white', alpha=0.7))
        
        plt.suptitle(f"Primary Gradient Axis: {gradient_info['primary_axis']}")
    
    plt.tight_layout()
    return fig