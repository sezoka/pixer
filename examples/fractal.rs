use pixer as pix;

#[derive(Clone, Copy)]

struct Complex {
    pub a: f32,
    pub b: f32,
}

impl std::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Complex {
            a: self.a + rhs.a,

            b: self.b + rhs.b,
        }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Complex {
            a: self.a * rhs.a - self.b * rhs.b,

            b: self.a * rhs.b + self.b * rhs.a,
        }
    }
}

impl Complex {
    fn arg_sq(self) -> f32 {
        self.a * self.a + self.b * self.b
    }
}

impl Complex {
    fn abs(self) -> Self {
        Complex {
            a: self.a.abs(),

            b: self.b.abs(),
        }
    }
}

fn mandelbrot(x: f32, y: f32, s: f32) -> f32 {
    let mut z = Complex { a: 0.0, b: 0.0 };

    let c = Complex { a: x * s, b: y * s };

    let max = 256;

    let mut i = 0;

    while i < max && z.arg_sq() < 32.0 {
        z = z * z + c;

        i += 1;
    }

    return (i as f32 - z.arg_sq().log2().log2()) / (max as f32);
}

fn julia(x: f32, y: f32) -> f32 {
    let mut z = Complex { a: x, b: y };

    let c = Complex { a: 0.38, b: 0.28 };

    let max = 256;

    let mut i = 0;

    while i < max && z.arg_sq() < 32.0 {
        z = z * z + c;

        i += 1;
    }

    return (i as f32 - z.arg_sq().log2().log2()) / (max as f32);
}

fn color(t: f32) -> (u8, u8, u8) {
    let a = (0.5, 0.5, 0.5);

    let b = (0.5, 0.5, 0.5);

    let c = (1.0, 1.0, 1.0);

    let d = (0.0, 0.10, 0.20);

    let r = b.0 * (6.28318 * (c.0 * t + d.0)).cos() + a.0;

    let g = b.1 * (6.28318 * (c.1 * t + d.1)).cos() + a.1;

    let b = b.2 * (6.28318 * (c.2 * t + d.2)).cos() + a.2;

    ((255.0 * r) as u8, (255.0 * g) as u8, (255.0 * b) as u8)
}

fn main() -> pix::Result<()> {
    let grid_width = 250;
    let grid_height = 250;

    let win_cfg = pix::WindowConfig::new()
        .title("Test window")
        .size(1, 1)
        .vsync()
        .resizable();
    let grid_cfg = pix::GridConfig::new().size(grid_width, grid_height);
    let mut pixer = pix::Pixer::new(&win_cfg, &grid_cfg)?;

    pixer.draw().clear();

    let mut vx = 0.0;
    let mut vy = 0.0;

    loop {
        while let Some(event) = pixer.poll_event() {
            if event == pix::Event::Quit {
                return Ok(());
            }

            pixer.update_keyboard_state();

            if pixer.keyboard().holded(sdl2::keyboard::Keycode::A) {
                vx -= 0.1;
            }

            if pixer.keyboard().holded(sdl2::keyboard::Keycode::D) {
                vx += 0.1;
            }

            if pixer.keyboard().holded(sdl2::keyboard::Keycode::W) {
                vy -= 0.1;
            }

            if pixer.keyboard().holded(sdl2::keyboard::Keycode::S) {
                vy += 0.1;
            }
        }

        for y in 0..grid_height {
            for x in 0..grid_width {
                let u = x as f32 / grid_width as f32;
                let v = y as f32 / grid_height as f32;
                let t = julia(2.5 * (u - 0.5) + vx, 2.5 * (v - 0.5) + vy);

                pixer
                    .draw()
                    .pixel((x as f32, y as f32), color((2.0 * t + 0.5) % 1.0));
            }
        }

        pixer.draw().present()
    }
}
