use nalgebra::{Vector2, Vector3, Matrix2, Matrix3, UnitQuaternion};

/// A 3D Gaussian splat with gradient tracking
#[derive(Debug, Clone)]
pub struct Gaussian3D {
    /// Position (mean μ)
    pub mean: Vector3<f32>,
    /// Scale factors (sx, sy, sz)
    pub scale: Vector3<f32>,
    /// Rotation (as quaternion)
    pub rotation: UnitQuaternion<f32>,
    /// Opacity (α)
    pub opacity: f32,
    /// RGB color
    pub color: Vector3<f32>,
    
    // Gradients
    pub grad_mean: Vector3<f32>,
    pub grad_scale: Vector3<f32>,
    pub grad_rotation: Vector3<f32>,
    pub grad_opacity: f32,
    pub grad_color: Vector3<f32>,
}

impl Gaussian3D {
    pub fn new(
        mean: Vector3<f32>,
        scale: Vector3<f32>,
        rotation: UnitQuaternion<f32>,
        opacity: f32,
        color: Vector3<f32>,
    ) -> Self {
        Self {
            mean,
            scale,
            rotation,
            opacity,
            color,
            grad_mean: Vector3::zeros(),
            grad_scale: Vector3::zeros(),
            grad_rotation: Vector3::zeros(),
            grad_opacity: 0.0,
            grad_color: Vector3::zeros(),
        }
    }

    /// Reset gradients to zero
    pub fn zero_grad(&mut self) {
        self.grad_mean = Vector3::zeros();
        self.grad_scale = Vector3::zeros();
        self.grad_rotation = Vector3::zeros();
        self.grad_opacity = 0.0;
        self.grad_color = Vector3::zeros();
    }

    /// Compute 3D covariance matrix
    pub fn covariance_3d(&self) -> Matrix3<f32> {
        let r = self.rotation.to_rotation_matrix();
        let s = Matrix3::from_diagonal(&self.scale);
        r.matrix() * s * s.transpose() * r.matrix().transpose()
    }
}

#[derive(Debug, Clone)]
pub struct Gaussian2D {
    /// 2D position (screen coordinates)
    pub position: Vector2<f32>,
    /// 2D covariance matrix
    pub covariance: Matrix2<f32>,
    /// Opacity
    pub opacity: f32,
    /// RGB color
    pub color: Vector3<f32>,
    /// Depth (for sorting)
    pub depth: f32,
    pub source_idx: usize,
}