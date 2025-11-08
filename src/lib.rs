pub mod core;
pub mod image;
pub mod render; 

// Re-exports for convenience
pub use core::gaussian::{Gaussian3D, Gaussian2D};
pub use core::camera::Camera;
pub use render::Rasterizer;
pub use image::ppm::PPMImage;
