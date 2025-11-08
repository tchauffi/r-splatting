//! Simple backward pass for gradient computation

use nalgebra::{Vector2, Vector3, Matrix2, Matrix3};
use crate::core::gaussian::{Gaussian3D, Gaussian2D};
use crate::image::ImageBuffer;

/// Backward pass - propagate gradients to position, color, opacity
pub fn backward_pass_simple(
    gaussians: &mut [Gaussian3D],
    gaussians_2d: &[Gaussian2D],
    grad_image: &ImageBuffer,
    width: usize,
    height: usize,
) {
    // Zero gradients
    for g in gaussians.iter_mut() {
        g.zero_grad();
    }
    
    // For each pixel, propagate gradient through alpha compositing
    for y in 0..height {
        for x in 0..width {
            let px = (x as f32 / width as f32) * 2.0 - 1.0;
            let py = 1.0 - (y as f32 / height as f32) * 2.0;
            let point = Vector2::new(px, py);
            
            let grad_color = grad_image.get_pixel(x, y);
            
            if grad_color.norm() < 1e-6 {
                continue;
            }
            
            // Forward pass to compute all alphas and transmittance values
            let mut alphas = Vec::with_capacity(gaussians_2d.len());
            let mut transmittances = Vec::with_capacity(gaussians_2d.len());
            let mut transmit = 1.0f32;
            
            for g2d in gaussians_2d {
                let alpha = evaluate_gaussian_2d(g2d, point);
                alphas.push(alpha);
                transmittances.push(transmit);
                transmit *= 1.0 - alpha;
                if transmit < 0.001 {
                    break;
                }
            }
            
            // Backward pass: compute gradients
            for (i, g2d) in gaussians_2d.iter().enumerate() {
                if i >= alphas.len() {
                    break;
                }
                
                let alpha = alphas[i];
                let transmit = transmittances[i];
                
                if alpha < 1e-4 {
                    continue;
                }
                
                let source = &mut gaussians[g2d.source_idx];
                
                // ∂L/∂color: gradient flows through color blending
                // C_out = T * α * c + ...
                let grad_c = grad_color * transmit * alpha;
                source.grad_color += grad_c;
                
                // ∂L/∂α: gradient flows through both color and transmittance
                // Color contribution: T * c
                let grad_alpha_from_color = grad_color.dot(&g2d.color) * transmit;
                
                // Transmittance contribution: affects all subsequent Gaussians
                let mut grad_alpha_from_transmit = 0.0f32;
                for j in (i + 1)..alphas.len() {
                    if j < gaussians_2d.len() {
                        let g2d_later = &gaussians_2d[j];
                        let alpha_later = alphas[j];
                        let transmit_later = transmittances[j];
                        
                        // T_j = T_i * (1 - α_i), so ∂T_j/∂α_i = -T_i
                        let contrib = grad_color.dot(&g2d_later.color) * alpha_later * (-transmit_later / (1.0 - alpha).max(1e-6));
                        grad_alpha_from_transmit += contrib;
                    }
                }
                
                let grad_alpha_total = grad_alpha_from_color + grad_alpha_from_transmit;
                
                // ∂α/∂opacity = G(x; μ, Σ) where G is the Gaussian value without opacity
                let gaussian_value = alpha / g2d.opacity.max(1e-6);
                source.grad_opacity += grad_alpha_total * gaussian_value;
                
                // ∂α/∂μ: gradient w.r.t. position
                let grad_pos_2d = compute_position_gradient(g2d, point, gaussian_value);
                let grad_pos_scaled = grad_pos_2d * grad_alpha_total * g2d.opacity;
                
                // Project back to 3D (just xy for orthographic)
                source.grad_mean.x += grad_pos_scaled.x;
                source.grad_mean.y += grad_pos_scaled.y;
                
                // ∂α/∂Σ: gradient w.r.t. covariance (and thus scale and rotation)
                let grad_cov = compute_covariance_gradient(g2d, point, gaussian_value);
                let grad_cov_scaled = grad_cov * grad_alpha_total * g2d.opacity;
                
                // According to the 3DGS paper, the gradient computation is:
                // Σ_3D = M * M^T where M = S * R
                // S = diag(sx, sy, sz), R = rotation matrix from quaternion
                // For orthographic: Σ_2D = upper-left 2x2 of Σ_3D
                
                // ∂L/∂Σ_3D: Convert 2D covariance gradient to 3D (pad with zeros)
                // dL_dSigma is symmetric 3x3 matrix
                let dl_dsigma_00 = grad_cov_scaled[(0, 0)];
                let dl_dsigma_01 = grad_cov_scaled[(0, 1)] * 0.5; // Off-diagonal appears twice
                let dl_dsigma_11 = grad_cov_scaled[(1, 1)];
                // Other elements are zero for orthographic projection
                
                // Compute M = S * R
                let rot_matrix_3d = source.rotation.to_rotation_matrix();
                let rot = rot_matrix_3d.matrix();
                
                let s = Vector3::new(source.scale.x, source.scale.y, source.scale.z);
                
                // M = S * R (3x3 matrix)
                let m = Matrix3::new(
                    s.x * rot[(0,0)], s.x * rot[(0,1)], s.x * rot[(0,2)],
                    s.y * rot[(1,0)], s.y * rot[(1,1)], s.y * rot[(1,2)],
                    s.z * rot[(2,0)], s.z * rot[(2,1)], s.z * rot[(2,2)],
                );
                
                // ∂Σ/∂M = 2*M (since Σ = M * M^T)
                // dL_dM = 2 * M * dL_dSigma
                let dl_dsigma_3d = Matrix3::new(
                    dl_dsigma_00, dl_dsigma_01, 0.0,
                    dl_dsigma_01, dl_dsigma_11, 0.0,
                    0.0, 0.0, 0.0,
                );
                
                // ∂Σ/∂M: Since Σ = M * M^T, we have ∂Σ/∂M = 2*M when Σ is symmetric
                // So dL/dM = 2 * dL/dΣ * M (chain rule)
                let dl_dm = (dl_dsigma_3d * m) * 2.0;
                
                // ∂L/∂scale: M = S * R, so ∂M/∂s_i = e_i * R (where e_i is i-th basis vector)
                // dL/ds_i = sum_j( dL/dM_ij * ∂M_ij/∂s_i ) = sum_j( dL/dM_ij * R_ij )
                // = dot(R[i,:], dL_dM[i,:])
                let rot_t = rot.transpose();
                source.grad_scale.x += rot_t[(0,0)] * dl_dm[(0,0)] + rot_t[(0,1)] * dl_dm[(0,1)] + rot_t[(0,2)] * dl_dm[(0,2)];
                source.grad_scale.y += rot_t[(1,0)] * dl_dm[(1,0)] + rot_t[(1,1)] * dl_dm[(1,1)] + rot_t[(1,2)] * dl_dm[(1,2)];
                source.grad_scale.z += rot_t[(2,0)] * dl_dm[(2,0)] + rot_t[(2,1)] * dl_dm[(2,1)] + rot_t[(2,2)] * dl_dm[(2,2)];
                
                // ∂L/∂rotation: M = S * R, so ∂M/∂R = S
                // Need to convert rotation gradient from matrix to quaternion
                // dl_dM_scaled = dL_dM * S (element-wise scaling of columns)
                let dl_dm_scaled = Matrix3::new(
                    dl_dm[(0,0)] * s.x, dl_dm[(0,1)] * s.y, dl_dm[(0,2)] * s.z,
                    dl_dm[(1,0)] * s.x, dl_dm[(1,1)] * s.y, dl_dm[(1,2)] * s.z,
                    dl_dm[(2,0)] * s.x, dl_dm[(2,1)] * s.y, dl_dm[(2,2)] * s.z,
                );
                
                // Gradient of rotation matrix w.r.t. quaternion components
                // From the paper: convert dL_dR to dL_dq using quaternion derivatives
                // This is a simplified approximation - full derivation is complex
                let r = source.rotation.as_ref()[0];
                let x = source.rotation.as_ref()[1];
                let y = source.rotation.as_ref()[2];
                let z = source.rotation.as_ref()[3];
                
                let dl_dmt = dl_dm_scaled.transpose();
                
                // Gradient w.r.t. quaternion (from CUDA code in backward.cu lines 330-336)
                let dq_w = 2.0 * z * (dl_dmt[(0,1)] - dl_dmt[(1,0)]) 
                         + 2.0 * y * (dl_dmt[(2,0)] - dl_dmt[(0,2)]) 
                         + 2.0 * x * (dl_dmt[(1,2)] - dl_dmt[(2,1)]);
                         
                let dq_x = 2.0 * y * (dl_dmt[(1,0)] + dl_dmt[(0,1)]) 
                         + 2.0 * z * (dl_dmt[(2,0)] + dl_dmt[(0,2)]) 
                         + 2.0 * r * (dl_dmt[(1,2)] - dl_dmt[(2,1)]) 
                         - 4.0 * x * (dl_dmt[(2,2)] + dl_dmt[(1,1)]);
                         
                let dq_y = 2.0 * x * (dl_dmt[(1,0)] + dl_dmt[(0,1)]) 
                         + 2.0 * r * (dl_dmt[(2,0)] - dl_dmt[(0,2)]) 
                         + 2.0 * z * (dl_dmt[(1,2)] + dl_dmt[(2,1)]) 
                         - 4.0 * y * (dl_dmt[(2,2)] + dl_dmt[(0,0)]);
                         
                let dq_z = 2.0 * r * (dl_dmt[(0,1)] - dl_dmt[(1,0)]) 
                         + 2.0 * x * (dl_dmt[(2,0)] + dl_dmt[(0,2)]) 
                         + 2.0 * y * (dl_dmt[(1,2)] + dl_dmt[(2,1)]) 
                         - 4.0 * z * (dl_dmt[(1,1)] + dl_dmt[(0,0)]);
                
                // Accumulate quaternion gradient (scaled for stability)
                // Use small factor to avoid large updates to rotation
                let quat_grad_scale = 0.1;
                source.grad_rotation.x += dq_x * quat_grad_scale;
                source.grad_rotation.y += dq_y * quat_grad_scale;
                source.grad_rotation.z += dq_z * quat_grad_scale;
            }
        }
    }
}

fn evaluate_gaussian_2d(gaussian: &Gaussian2D, point: Vector2<f32>) -> f32 {
    let diff = point - gaussian.position;
    
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
    
    let mahalanobis = diff.transpose() * inv_cov * diff;
    let value = (-0.5 * mahalanobis[(0, 0)]).exp();
    
    value * gaussian.opacity
}

fn compute_position_gradient(gaussian: &Gaussian2D, point: Vector2<f32>, gaussian_value: f32) -> Vector2<f32> {
    let diff = point - gaussian.position;
    
    let det = gaussian.covariance[(0, 0)] * gaussian.covariance[(1, 1)]
        - gaussian.covariance[(0, 1)] * gaussian.covariance[(1, 0)];
    
    if det.abs() < 1e-6 {
        return Vector2::zeros();
    }
    
    let inv_cov = Matrix2::new(
        gaussian.covariance[(1, 1)] / det,
        -gaussian.covariance[(0, 1)] / det,
        -gaussian.covariance[(1, 0)] / det,
        gaussian.covariance[(0, 0)] / det,
    );
    
    // ∂G/∂μ = G * Σ^(-1) * (μ - x) = -G * Σ^(-1) * (x - μ)
    -(inv_cov * diff) * gaussian_value
}

fn compute_covariance_gradient(gaussian: &Gaussian2D, point: Vector2<f32>, gaussian_value: f32) -> Matrix2<f32> {
    let diff = point - gaussian.position;
    
    let det = gaussian.covariance[(0, 0)] * gaussian.covariance[(1, 1)]
        - gaussian.covariance[(0, 1)] * gaussian.covariance[(1, 0)];
    
    if det.abs() < 1e-6 {
        return Matrix2::zeros();
    }
    
    let inv_cov = Matrix2::new(
        gaussian.covariance[(1, 1)] / det,
        -gaussian.covariance[(0, 1)] / det,
        -gaussian.covariance[(1, 0)] / det,
        gaussian.covariance[(0, 0)] / det,
    );
    
    // ∂G/∂Σ = -0.5 * G * Σ^(-1) * [(x-μ)(x-μ)^T * Σ^(-1) - Σ^(-1)]
    let mahalanobis_vec = inv_cov * diff;
    let outer_product = Matrix2::new(
        diff.x * mahalanobis_vec.x, diff.x * mahalanobis_vec.y,
        diff.y * mahalanobis_vec.x, diff.y * mahalanobis_vec.y,
    );
    
    -0.5 * gaussian_value * (outer_product - inv_cov)
}