//! Loss functions: L1 + D-SSIM with PyTorch tensors

use tch::Tensor;
use crate::image::ImageBuffer;

#[derive(Debug, Clone)]
pub struct LossMetrics {
    pub l1_loss: f32,
    pub ssim_loss: f32,
    pub total_loss: f32,
}

/// Convert ImageBuffer to PyTorch tensor [H, W, 3]
pub fn image_to_tensor(image: &ImageBuffer) -> Tensor {
    let height = image.height as i64;
    let width = image.width as i64;
    
    // Flatten pixels to Vec<f32>
    let mut data = Vec::with_capacity(image.pixels.len() * 3);
    for pixel in &image.pixels {
        data.push(pixel.x);
        data.push(pixel.y);
        data.push(pixel.z);
    }
    
    Tensor::from_slice(&data)
        .view([height, width, 3])
}

/// Compute L1 loss between rendered and ground truth using PyTorch
pub fn compute_l1_loss_torch(rendered: &Tensor, ground_truth: &Tensor) -> Tensor {
    (rendered - ground_truth).abs().mean(tch::Kind::Float)
}

/// Compute L1 loss (Rust version for compatibility)
pub fn compute_l1_loss(rendered: &ImageBuffer, ground_truth: &ImageBuffer) -> f32 {
    assert_eq!(rendered.width, ground_truth.width);
    assert_eq!(rendered.height, ground_truth.height);
    
    let mut loss = 0.0;
    let n = (rendered.width * rendered.height) as f32;
    
    for i in 0..rendered.pixels.len() {
        let diff = rendered.pixels[i] - ground_truth.pixels[i];
        loss += diff.abs().sum();
    }
    
    loss / (n * 3.0)
}

/// Compute SSIM using PyTorch tensors
pub fn compute_ssim_torch(rendered: &Tensor, ground_truth: &Tensor) -> Tensor {
    // Constants for SSIM
    let c1 = 0.01_f64.powi(2);
    let c2 = 0.03_f64.powi(2);
    
    // Compute means
    let mean_x = rendered.mean(tch::Kind::Float);
    let mean_y = ground_truth.mean(tch::Kind::Float);
    
    // Compute variances and covariance
    let diff_x = rendered - &mean_x;
    let diff_y = ground_truth - &mean_y;
    
    let var_x = (&diff_x * &diff_x).mean(tch::Kind::Float);
    let var_y = (&diff_y * &diff_y).mean(tch::Kind::Float);
    let cov_xy = (&diff_x * &diff_y).mean(tch::Kind::Float);
    
    // SSIM formula
    let numerator = (2.0 * &mean_x * &mean_y + c1) * (2.0 * cov_xy + c2);
    let denominator = (mean_x.pow_tensor_scalar(2) + mean_y.pow_tensor_scalar(2) + c1) 
        * (var_x + var_y + c2);
    
    numerator / denominator
}

/// Compute SSIM (Rust version for compatibility)
pub fn compute_ssim(rendered: &ImageBuffer, ground_truth: &ImageBuffer) -> f32 {
    let c1 = 0.01_f32.powi(2);
    let c2 = 0.03_f32.powi(2);
    
    let mut mean_x = nalgebra::Vector3::zeros();
    let mut mean_y = nalgebra::Vector3::zeros();
    let n = rendered.pixels.len() as f32;
    
    for i in 0..rendered.pixels.len() {
        mean_x += rendered.pixels[i];
        mean_y += ground_truth.pixels[i];
    }
    mean_x /= n;
    mean_y /= n;
    
    let mut var_x = nalgebra::Vector3::zeros();
    let mut var_y = nalgebra::Vector3::zeros();
    let mut cov_xy = nalgebra::Vector3::zeros();
    
    for i in 0..rendered.pixels.len() {
        let dx = rendered.pixels[i] - mean_x;
        let dy = ground_truth.pixels[i] - mean_y;
        var_x += dx.component_mul(&dx);
        var_y += dy.component_mul(&dy);
        cov_xy += dx.component_mul(&dy);
    }
    var_x /= n;
    var_y /= n;
    cov_xy /= n;
    
    let mut ssim = 0.0;
    for c in 0..3 {
        let numerator = (2.0 * mean_x[c] * mean_y[c] + c1) * (2.0 * cov_xy[c] + c2);
        let denominator = (mean_x[c].powi(2) + mean_y[c].powi(2) + c1) * (var_x[c] + var_y[c] + c2);
        ssim += numerator / denominator;
    }
    
    ssim / 3.0
}

/// Compute combined loss with PyTorch tensors: (1-λ)*L1 + λ*D-SSIM
pub fn compute_loss_torch(
    rendered: &Tensor,
    ground_truth: &Tensor,
    lambda: f64,
) -> Tensor {
    let l1 = compute_l1_loss_torch(rendered, ground_truth);
    let ssim = compute_ssim_torch(rendered, ground_truth);
    let d_ssim = 1.0 - ssim; // D-SSIM = 1 - SSIM
    
    (1.0 - lambda) * l1 + lambda * d_ssim
}

/// Compute combined loss (Rust version): (1-λ)*L1 + λ*D-SSIM
pub fn compute_loss(
    rendered: &ImageBuffer,
    ground_truth: &ImageBuffer,
    lambda: f32,
) -> LossMetrics {
    let l1 = compute_l1_loss(rendered, ground_truth);
    let ssim = compute_ssim(rendered, ground_truth);
    let d_ssim = 1.0 - ssim;
    
    let total = (1.0 - lambda) * l1 + lambda * d_ssim;
    
    LossMetrics {
        l1_loss: l1,
        ssim_loss: d_ssim,
        total_loss: total,
    }
}

/// Compute gradient of loss w.r.t. rendered image (L1 only for stability)
pub fn compute_loss_gradient(
    rendered: &ImageBuffer,
    ground_truth: &ImageBuffer,
    _lambda: f32, // Ignored - using L1 only
) -> ImageBuffer {
    let mut gradient = ImageBuffer::new(rendered.width, rendered.height);
    
    // L1 gradient: sign(rendered - ground_truth)
    for i in 0..rendered.pixels.len() {
        let diff = rendered.pixels[i] - ground_truth.pixels[i];
        
        let grad_l1 = nalgebra::Vector3::new(
            if diff.x > 0.0 { 1.0 } else if diff.x < 0.0 { -1.0 } else { 0.0 },
            if diff.y > 0.0 { 1.0 } else if diff.y < 0.0 { -1.0 } else { 0.0 },
            if diff.z > 0.0 { 1.0 } else if diff.z < 0.0 { -1.0 } else { 0.0 },
        );
        
        // Use L1 gradient only
        gradient.pixels[i] = grad_l1;
    }
    
    gradient
}

/// Compute gradient using PyTorch autograd
pub fn compute_loss_gradient_torch(
    rendered: &Tensor,
    ground_truth: &Tensor,
    lambda: f64,
) -> Tensor {
    // Enable gradient computation
    let rendered = rendered.set_requires_grad(true);
    
    // Compute loss
    let loss = compute_loss_torch(&rendered, ground_truth, lambda);
    
    // Backward pass
    loss.backward();
    
    // Get gradient
    rendered.grad()
}