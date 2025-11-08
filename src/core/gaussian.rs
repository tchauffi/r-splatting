use nalgebra::{Vector2, Vector3, Matrix2, Matrix3, UnitQuaternion};

#[derive(Debug, Clone)]
pub struct Gaussian3D {
    pub mean: Vector3<f32>,
    pub scale: Vector3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub color: Vector3<f32>,
    pub opacity: f32,
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
        }
    }

    /// Compute the 3D covariance matrix Σ = R S S^T R^T
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
}