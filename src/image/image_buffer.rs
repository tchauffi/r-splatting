//! Image buffer for storing and manipulating images

use std::io::Result;
use nalgebra::Vector3;

pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<Vector3<f32>>, // RGB float [0, 1]
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![Vector3::zeros(); width * height],
        }
    }

    /// Load image from file (JPG, PNG, etc.)
    pub fn load(path: &str) -> Result<Self> {
        let img = image::open(path)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?
            .to_rgb8();
        
        let (width, height) = img.dimensions();
        let width = width as usize;
        let height = height as usize;
        
        let mut pixels = Vec::with_capacity(width * height);
        
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x as u32, y as u32);
                pixels.push(Vector3::new(
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                ));
            }
        }
        
        Ok(Self {
            width,
            height,
            pixels,
        })
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

    /// Write image to PNG
    pub fn save(&self, filename: &str) -> Result<()> {
        let mut img_buffer = image::RgbImage::new(self.width as u32, self.height as u32);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.get_pixel(x, y);
                img_buffer.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgb([
                        (pixel.x.clamp(0.0, 1.0) * 255.0) as u8,
                        (pixel.y.clamp(0.0, 1.0) * 255.0) as u8,
                        (pixel.z.clamp(0.0, 1.0) * 255.0) as u8,
                    ]),
                );
            }
        }
        
        img_buffer.save(filename)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}