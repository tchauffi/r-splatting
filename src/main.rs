use nalgebra::{Vector3, UnitQuaternion, Matrix4};
use r_splatting::{Gaussian3D, Rasterizer};

fn main() {
    println!("Creating simple Gaussian Splatting scene...");
    
    // Create a few Gaussians
    let mut gaussians = Vec::new();
    
    // Red Gaussian
    gaussians.push(Gaussian3D::new(
        Vector3::new(-0.5, 0.0, -2.0),
        
        Vector3::new(0.3, 0.3, 0.3),
        UnitQuaternion::identity(),
        0.8,
        Vector3::new(1.0, 0.0, 0.0), // Red
    ));
    
    // Green Gaussian
    gaussians.push(Gaussian3D::new(
        Vector3::new(0.5, 0.0, -2.0),
        Vector3::new(0.3, 0.3, 0.3),
        UnitQuaternion::identity(),
        0.8,
        Vector3::new(0.0, 1.0, 0.0), // Green
    ));
    
    // Blue Gaussian (in front)
    gaussians.push(Gaussian3D::new(
        Vector3::new(0.0, 0.3, -1.5),
        Vector3::new(0.4, 0.4, 0.4),
        UnitQuaternion::identity(),
        0.6,
        Vector3::new(0.0, 0.0, 1.0), // Blue
    ));
    
    // Create rasterizer
    let rasterizer = Rasterizer::new(800, 600);
    
    // Simple view matrix (identity for now)
    let view_matrix = Matrix4::identity();
    
    // Render
    println!("Rendering...");
    let image = rasterizer.render(&gaussians, &view_matrix);
    
    // Save to PPM
    println!("Writing output.ppm...");
    image.write_ppm("output.ppm").expect("Failed to write image");
    
    println!("Done! View output.ppm");
}