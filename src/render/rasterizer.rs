//! CPU-based Gaussian rasterizer

use nalgebra::{Vector2, Vector3, Matrix2, Matrix4};
use crate::core::gaussian::{Gaussian2D, Gaussian3D};
use crate::core::projection::project_gaussian_orthographic;
use crate::image::PPMImage;

pub struct Rasterizer {
    width: usize,
    height: usize,
}

impl Rasterizer {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Evaluate 2D Gaussian at a point
    fn evaluate_gaussian_2d(
        gaussian: &Gaussian2D,
        point: Vector2<f32>,
    ) -> f32 {
        let diff = point - gaussian.position;
        
        // Compute inverse covariance
        let det = gaussian.covariance[(0, 0)] * gaussian.covariance[(1, 1)]
            - gaussian.covariance[(0, 1)] * gaussian.covariance[(1, 0)];
        
        if det.abs() < 1e-6 {
            return 0.0;
        }
        
        let inv_cov = Matrix2::new(
            gaussian.covariance[(1, 1)] / det,
            -gaussian.covariance[(0, 1)] / det,
            -gaussian.covariance[(1, 0)] / det,
            gaussian.covariance[(0, 0)] / det,
        );
        
        // Mahalanobis distance: d^2 = (x-μ)^T Σ^-1 (x-μ)
        let mahalanobis = diff.transpose() * inv_cov * diff;
        
        // Gaussian value: exp(-0.5 * d^2)
        let value = (-0.5 * mahalanobis[(0, 0)]).exp();
        
        value * gaussian.opacity
    }

    /// Render scene of 3D Gaussians to image
    pub fn render(
        &self,
        gaussians: &[Gaussian3D],
        view_matrix: &Matrix4<f32>,
    ) -> PPMImage {
        let mut image = PPMImage::new(self.width, self.height);
        
        // Project all Gaussians to 2D
        let mut gaussians_2d: Vec<Gaussian2D> = gaussians
            .iter()
            .filter_map(|g| project_gaussian_orthographic(g, view_matrix))
            .collect();
        
        // Sort by depth (back to front for now - we'll do front-to-back later)
        gaussians_2d.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
        
        // For each pixel
        for y in 0..self.height {
            for x in 0..self.width {
                // Convert pixel to world coordinates [-1, 1]
                let px = (x as f32 / self.width as f32) * 2.0 - 1.0;
                let py = 1.0 - (y as f32 / self.height as f32) * 2.0; // Flip Y
                let point = Vector2::new(px, py);
                
                let mut color = Vector3::zeros();
                let mut alpha = 0.0f32;
                
                // Alpha blend all Gaussians at this pixel (back to front)
                for gaussian in &gaussians_2d {
                    let weight = Self::evaluate_gaussian_2d(gaussian, point);
                    
                    if weight > 1e-4 {
                        // Standard alpha blending: C = C_old * (1 - α) + C_new * α
                        color = color * (1.0 - weight) + gaussian.color * weight;
                        alpha = alpha + (1.0 - alpha) * weight;
                        
                        // Early termination if fully opaque
                        if alpha > 0.999 {
                            break;
                        }
                    }
                }
                
                image.set_pixel(x, y, color);
            }
        }
        
        image
    }
}