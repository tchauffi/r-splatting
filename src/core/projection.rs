use nalgebra::{Matrix4, Vector2, Vector3, Matrix2};
use crate::core::gaussian::{Gaussian2D, Gaussian3D};

/// Simple orthographic projection (we'll add perspective later);
pub fn project_gaussian_orthographic(
    gaussian: &Gaussian3D,
    view_matrix: &Matrix4<f32>,
) -> Option<Gaussian2D> {
    // Transform position to view space
    let pos_homogeneous = view_matrix * gaussian.mean.to_homogeneous();
    let pos_view = Vector3::new(pos_homogeneous.x, pos_homogeneous.y, pos_homogeneous.z);
    
    // Cull if behind camera
    if pos_view.z > 0.0 {
        return None;
    }
    
    // Project to 2D (simple orthographic: just drop z)
    let pos_2d = Vector2::new(pos_view.x, pos_view.y);
    
    // Simplified 2D covariance projection
    // For orthographic: just take upper-left 2x2 of 3D covariance
    let cov_3d = gaussian.covariance_3d();
    let cov_2d = Matrix2::new(
        cov_3d[(0, 0)], cov_3d[(0, 1)],
        cov_3d[(1, 0)], cov_3d[(1, 1)],
    );
    
    Some(Gaussian2D {
        position: pos_2d,
        covariance: cov_2d,
        opacity: gaussian.opacity,
        color: gaussian.color,
        depth: pos_view.z,
    })
}