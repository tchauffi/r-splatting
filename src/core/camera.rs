use nalgebra::{Matrix4, Point3, Vector3};

pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,
    pub fov: f32,  // Field of view in radians
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Camera {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 3.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
            fov: std::f32::consts::FRAC_PI_4, // 45 degrees
            aspect: width as f32 / height as f32,
            near: 0.1,
            far: 100.0,
        }
    }

    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.position, &self.target, &self.up)
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        Matrix4::new_perspective(self.aspect, self.fov, self.near, self.far)
    }
}