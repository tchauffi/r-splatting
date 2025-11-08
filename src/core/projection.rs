use nalgebra::{Matrix4, Vector2, Vector3, Matrix2, Matrix3};
use crate::core::gaussian::{Gaussian2D, Gaussian3D};

/// Project 3D Gaussian to 2D with perspective projection
pub fn project_gaussian_perspective(
    gaussian: &Gaussian3D,
    view_matrix: &Matrix4<f32>,
    proj_matrix: &Matrix4<f32>,
    viewport_width: f32,
    viewport_height: f32,
) -> Option<Gaussian2D> {
    // Transform to view space
    let pos_view = view_matrix.transform_point(&gaussian.mean.into());
    
    // Cull if behind camera
    if pos_view.z >= 0.0 {
        return None;
    }
    
    // Project to clip space
    let pos_clip = proj_matrix.transform_point(&pos_view);
    
    // Perspective divide
    let pos_ndc = Vector3::new(
        pos_clip.x / pos_clip.z,
        pos_clip.y / pos_clip.z,
        pos_clip.z,
    );
    
    // Convert to screen space
    let pos_2d = Vector2::new(
        pos_ndc.x,
        pos_ndc.y,
    );
    
    // Compute Jacobian of projection for covariance
    let focal = viewport_height / (2.0 * (std::f32::consts::FRAC_PI_4 / 2.0).tan());
    let tan_fov = (std::f32::consts::FRAC_PI_4 / 2.0).tan();
    let lim_x = 1.3 * tan_fov;
    let lim_y = 1.3 * tan_fov;
    
    let tx = (pos_view.x / pos_view.z).clamp(-lim_x, lim_x) * pos_view.z;
    let ty = (pos_view.y / pos_view.z).clamp(-lim_y, lim_y) * pos_view.z;
    let tz = pos_view.z;
    
    // Jacobian of perspective projection
    let j = Matrix3::new(
        focal / tz, 0.0, -(focal * tx) / (tz * tz),
        0.0, focal / tz, -(focal * ty) / (tz * tz),
        0.0, 0.0, 0.0,
    );
    
    // Transform covariance
    let cov_3d = gaussian.covariance_3d();
    let view_rot = view_matrix.fixed_view::<3, 3>(0, 0);
    let cov_view = view_rot * cov_3d * view_rot.transpose();
    
    let cov_2d_full = j * cov_view * j.transpose();
    let cov_2d = Matrix2::new(
        cov_2d_full[(0, 0)], cov_2d_full[(0, 1)],
        cov_2d_full[(1, 0)], cov_2d_full[(1, 1)],
    );
    
    Some(Gaussian2D {
        position: pos_2d,
        covariance: cov_2d,
        opacity: gaussian.opacity,
        color: gaussian.color,
        depth: -pos_view.z,
    })
}

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