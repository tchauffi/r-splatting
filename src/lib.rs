pub mod core;
pub mod image;
pub mod render; 
pub mod train;

// Re-exports for convenience
pub use core::gaussian::{Gaussian3D, Gaussian2D};
pub use core::camera::Camera;
pub use render::Rasterizer;
pub use image::ImageBuffer;
pub use train::loss::{compute_loss, LossMetrics};
