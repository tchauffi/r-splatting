use tch::Device;
use r_splatting::{Gaussian3D, Rasterizer, Camera, ImageBuffer};
use r_splatting::train::{compute_loss, compute_loss_gradient, backward_pass_simple};
use nalgebra::{Vector3, UnitQuaternion};

fn main() {
    println!("Training Gaussian Splatting with profile picture...");
    
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);
    
    // Load and resize profile picture to 128x128
    let ground_truth = load_and_resize_image("image.png", 128, 128)
        .expect("Failed to load and resize image.png");
    
    println!("Loaded image: {}x{}", ground_truth.width, ground_truth.height);
    
    // Save target for reference
    ground_truth.save("target.png").expect("Failed to write target");
    
    // Get image dimensions
    let width = ground_truth.width;
    let height = ground_truth.height;
    
    // Initialize Gaussians
    let num_gaussians = 2000; // More Gaussians for complex image
    let mut gaussians = initialize_gaussians_grid(num_gaussians, width, height);
    
    // Create camera and rasterizer with matching dimensions
    let camera = Camera::new(width, height);
    let rasterizer = Rasterizer::new(width, height);
    
    let num_iterations = 1000;
    
    // Simplified learning rates - no momentum, cleaner debugging
    let lr_position_init: f32 = 0.00016;
    let lr_position_final: f32 = 0.0000016;
    let lr_position_max_steps = num_iterations;
    let spatial_lr_scale: f32 = 1.0;
    
    let lr_scale: f32 = 0.005;
    let lr_rotation: f32 = 0.001;
    let lr_opacity: f32 = 0.05;
    let lr_color: f32 = 0.0025;
    
    // Gradient clipping
    let grad_clip_max: f32 = 1.0;
    
    // Opacity reset interval
    let opacity_reset_interval = 300;
    
    for iter in 0..num_iterations {
        // Exponential learning rate decay for position
        let t = (iter as f32 / lr_position_max_steps as f32).min(1.0);
        let log_lerp = (lr_position_init.ln() * (1.0 - t) + lr_position_final.ln() * t).exp();
        let lr_position_current = log_lerp * spatial_lr_scale;
        
        // Render
        let (rendered, gaussians_2d) = rasterizer.render_with_gaussians_2d(
            &gaussians,
            &camera.view_matrix()
        );
        
        // Compute loss (L1 only)
        let loss_metrics = compute_loss(&rendered, &ground_truth, 0.0);
        
        // Compute gradient (L1 only)
        let grad_image = compute_loss_gradient(&rendered, &ground_truth, 0.0);
        
        // Backward pass
        backward_pass_simple(&mut gaussians, &gaussians_2d, &grad_image, width, height);
        
        // Simple gradient descent without momentum
        for g in gaussians.iter_mut() {
            // Clip gradients
            g.grad_mean = g.grad_mean.map(|x| x.clamp(-grad_clip_max, grad_clip_max));
            g.grad_scale = g.grad_scale.map(|x| x.clamp(-grad_clip_max, grad_clip_max));
            g.grad_rotation = g.grad_rotation.map(|x| x.clamp(-grad_clip_max, grad_clip_max));
            g.grad_opacity = g.grad_opacity.clamp(-grad_clip_max, grad_clip_max);
            g.grad_color = g.grad_color.map(|x| x.clamp(-grad_clip_max, grad_clip_max));
            
            // Update position
            g.mean -= lr_position_current * g.grad_mean;
            
            // Update scale
            g.scale -= lr_scale * g.grad_scale;
            g.scale = g.scale.map(|x| x.max(0.001).min(1.0));
            
            // Update rotation
            let rotation_magnitude = g.grad_rotation.norm();
            if rotation_magnitude > 1e-6 {
                use nalgebra::Unit;
                let axis = Unit::new_normalize(g.grad_rotation);
                let angle = -lr_rotation * rotation_magnitude;
                let delta_rotation = UnitQuaternion::from_axis_angle(&axis, angle);
                g.rotation = Unit::new_normalize((delta_rotation * g.rotation).into_inner());
            }
            
            // Update opacity
            g.opacity -= lr_opacity * g.grad_opacity;
            g.opacity = g.opacity.clamp(0.01, 0.99);
            
            // Update color
            g.color -= lr_color * g.grad_color;
            g.color = g.color.map(|x| x.clamp(0.0, 1.0));
        }
        
        // Opacity reset (from paper - prevents Gaussians from becoming too transparent/opaque)
        if iter > 0 && iter % opacity_reset_interval == 0 {
            for g in gaussians.iter_mut() {
                // Reset opacity to min(opacity, 0.01) to encourage re-learning
                g.opacity = g.opacity.min(0.01);
            }
            println!("  [Reset opacities at iteration {}]", iter);
        }
        
        // Logging
        if iter % 10 == 0 {
            println!(
                "Iteration {}: Loss = {:.6} (L1: {:.6}, SSIM: {:.6}), LR_pos = {:.8}",
                iter, loss_metrics.total_loss, loss_metrics.l1_loss, loss_metrics.ssim_loss, lr_position_current
            );
        }
        
        if iter % 100 == 0 {
            rendered.save(&format!("output_{:04}.png", iter))
                .expect("Failed to write image");
        }
    }
    
    println!("Training complete!");
}

/// Load image and resize to specified dimensions
fn load_and_resize_image(path: &str, target_width: usize, target_height: usize) -> std::io::Result<ImageBuffer> {
    use image::imageops::FilterType;
    
    let img = image::open(path)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    
    // Resize using Lanczos3 filter (high quality)
    let resized = img.resize_exact(
        target_width as u32,
        target_height as u32,
        FilterType::Lanczos3
    ).to_rgb8();
    
    let mut pixels = Vec::with_capacity(target_width * target_height);
    
    for y in 0..target_height {
        for x in 0..target_width {
            let pixel = resized.get_pixel(x as u32, y as u32);
            pixels.push(Vector3::new(
                pixel[0] as f32 / 255.0,
                pixel[1] as f32 / 255.0,
                pixel[2] as f32 / 255.0,
            ));
        }
    }
    
    Ok(ImageBuffer {
        width: target_width,
        height: target_height,
        pixels,
    })
}

// Initialize Gaussians in a grid pattern matching image dimensions
fn initialize_gaussians_grid(n: usize, width: usize, height: usize) -> Vec<Gaussian3D> {
    use rand::Rng;
    let mut rng = rand::rng();
    
    let aspect = width as f32 / height as f32;
    let grid_size = (n as f32).sqrt() as usize;
    
    let mut gaussians = Vec::new();
    
    for i in 0..grid_size {
        for j in 0..grid_size {
            if gaussians.len() >= n {
                break;
            }
            
            // Position in normalized coordinates [-1, 1]
            let x = (i as f32 / grid_size as f32) * 2.0 * aspect - aspect;
            let y = (j as f32 / grid_size as f32) * 2.0 - 1.0;
            
            // Add some randomness
            let x = x + rng.random_range(-0.1..0.1);
            let y = y + rng.random_range(-0.1..0.1);
            
            gaussians.push(Gaussian3D::new(
                Vector3::new(x, y, -2.0),
                Vector3::new(
                    rng.random_range(0.03..0.07),
                    rng.random_range(0.03..0.07),
                    0.05
                ), // Random initial scales to break symmetry
                UnitQuaternion::identity(),
                0.5, // Medium initial opacity
                Vector3::new(
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7),
                    rng.random_range(0.3..0.7)
                ), // Random initial colors to break symmetry
            ));
        }
    }
    
    // Fill remaining with random positions
    while gaussians.len() < n {
        gaussians.push(Gaussian3D::new(
            Vector3::new(
                rng.random_range(-aspect..aspect),
                rng.random_range(-1.0..1.0),
                -2.0,
            ),
            Vector3::new(0.05, 0.05, 0.05),
            UnitQuaternion::identity(),
            0.5,
            Vector3::new(
                rng.random::<f32>(),
                rng.random::<f32>(),
                rng.random::<f32>(),
            ),
        ));
    }
    
    gaussians
}