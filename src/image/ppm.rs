//! PPM image format writer

use std::fs::File;
use std::io::{Write, Result};
use nalgebra::Vector3;

pub struct PPMImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Vector3<f32>>, // RGB float [0, 1]
}

impl PPMImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![Vector3::zeros(); width * height],
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, color: Vector3<f32>) {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x] = color;
        }
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> Vector3<f32> {
        if x < self.width && y < self.height {
            self.pixels[y * self.width + x]
        } else {
            Vector3::zeros()
        }
    }

    /// Write image to PPM file
    pub fn write_ppm(&self, filename: &str) -> Result<()> {
        let mut file = File::create(filename)?;
        
        // PPM header
        writeln!(file, "P3")?;
        writeln!(file, "{} {}", self.width, self.height)?;
        writeln!(file, "255")?;
        
        // Write pixels
        for pixel in &self.pixels {
            let r = (pixel.x.clamp(0.0, 1.0) * 255.0) as u8;
            let g = (pixel.y.clamp(0.0, 1.0) * 255.0) as u8;
            let b = (pixel.z.clamp(0.0, 1.0) * 255.0) as u8;
            writeln!(file, "{} {} {}", r, g, b)?;
        }
        
        Ok(())
    }
}