//! CPU-based Gaussian rasterizer

use crate::core::gaussian::{Gaussian2D, Gaussian3D};
use crate::core::projection::project_gaussian_orthographic;
use crate::image::PPMImage;
use nalgebra::{Matrix2, Matrix4, Vector2, Vector3};

pub struct Rasterizer {
    width: usize,
    height: usize,
}

impl Rasterizer {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }

    /// Evaluate 2D Gaussian at a point
    fn evaluate_gaussian_2d(gaussian: &Gaussian2D, point: Vector2<f32>) -> f32 {
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
    pub fn render(&self, gaussians: &[Gaussian3D], view_matrix: &Matrix4<f32>) -> PPMImage {
        let mut image = PPMImage::new(self.width, self.height);

        // Project all Gaussians to 2D
        let mut gaussians_2d: Vec<Gaussian2D> = gaussians
            .iter()
            .filter_map(|g| project_gaussian_orthographic(g, view_matrix))
            .collect();

        // Sort front to back (negative depth = closer)
        gaussians_2d.sort_by(|a, b| b.depth.partial_cmp(&a.depth).unwrap());

        // Alpha blending (front to back)
        for y in 0..self.height {
            for x in 0..self.width {
                let px = (x as f32 / self.width as f32) * 2.0 - 1.0;
                let py = 1.0 - (y as f32 / self.height as f32) * 2.0;
                let point = Vector2::new(px, py);

                let mut color = Vector3::zeros();
                let mut transmittance = 1.0f32; // T = 1 initially

                // Front-to-back blending
                for gaussian in &gaussians_2d {
                    if transmittance < 0.001 {
                        break; // Early termination
                    }

                    let alpha = Self::evaluate_gaussian_2d(gaussian, point);

                    if alpha > 1e-4 {
                        // C_i = C_{i-1} + T_i * α_i * c_i
                        color += transmittance * alpha * gaussian.color;
                        // T_{i+1} = T_i * (1 - α_i)
                        transmittance *= 1.0 - alpha;
                    }
                }

                // Add background (white)
                color += transmittance * Vector3::new(1.0, 1.0, 1.0);

                image.set_pixel(x, y, color);
            }
        }

        image
    }
}
