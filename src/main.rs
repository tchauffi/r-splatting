use nalgebra::{Vector3, UnitQuaternion};
use r_splatting::{Gaussian3D, Rasterizer, Camera};

fn create_test_scene() -> Vec<Gaussian3D> {
    let mut gaussians = Vec::new();
    
    // Create a grid of Gaussians
    for i in -2..=2 {
        for j in -2..=2 {
            let x = i as f32 * 0.5;
            let y = j as f32 * 0.5;
            let z = -3.0 - (i * i + j * j) as f32 * 0.1;
            
            let color = Vector3::new(
                (i + 2) as f32 / 4.0,
                (j + 2) as f32 / 4.0,
                0.5,
            );
            
            gaussians.push(Gaussian3D::new(
                Vector3::new(x, y, z),
                Vector3::new(0.2, 0.2, 0.2),
                UnitQuaternion::identity(),
                1.0,
                color,
            ));
        }
    }
    
    gaussians
}

fn main() {
    println!("Creating simple Gaussian Splatting scene...");
    
    // Create scene
    let gaussians = create_test_scene();
    
    // Create camera
    let camera = Camera::new(800, 600);
    println!("Camera position: {:?}", camera.position);
    println!("Camera target: {:?}", camera.target);
    println!("FOV: {:.1}°", camera.fov.to_degrees());
    
    // Create rasterizer
    let rasterizer = Rasterizer::new(800, 600);
    
    // Get view matrix from camera
    let view_matrix = camera.view_matrix();
    
    // Render
    println!("Rendering...");
    let start = std::time::Instant::now();
    let image = rasterizer.render(&gaussians, &view_matrix);
    let duration = start.elapsed();
    println!("Render time: {:.3} ms", duration.as_secs_f64() * 1000.0);

    // Save to PPM
    println!("Writing output.ppm...");
    image.write_ppm("output.ppm").expect("Failed to write image");
    
    println!("Done! View output.ppm");
}