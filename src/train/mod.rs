//! Training components for Gaussian Splatting

pub mod loss;
pub mod backward;

pub use loss::{compute_loss, compute_loss_gradient, LossMetrics};
pub use backward::backward_pass_simple;