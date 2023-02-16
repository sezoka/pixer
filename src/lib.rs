use std::{
    collections::HashSet,
    error, fmt,
    mem::swap,
    result,
    time::{Duration, Instant},
};

use sdl2::{
    event::{Event as SDLEvent, WindowEvent},
    keyboard::Keycode,
    mouse::MouseButton as SDLMouseBtn,
    pixels::{self, PixelFormatEnum},
    rect::Rect as SDLRect,
    render::{Canvas, Texture, TextureCreator},
    video::{self, WindowContext},
    EventPump, Sdl,
};

pub const BYTES_PER_PIXEL: usize = 3;

pub type Result<T> = result::Result<T, Error>;

pub struct Point {
    x: f32,
    y: f32,
}

impl<T: Into<f32>> From<(T, T)> for Point {
    fn from(v: (T, T)) -> Self {
        Self {
            x: v.0.into(),
            y: v.1.into(),
        }
    }
}

pub struct Rect {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
}

impl<T: Into<f32>> From<(T, T, T, T)> for Rect {
    fn from(v: (T, T, T, T)) -> Self {
        Self {
            x: v.0.into(),
            y: v.1.into(),
            h: v.2.into(),
            w: v.3.into(),
        }
    }
}

#[derive(Debug)]
pub enum Error {
    InvalidWindowSize,
    InvalidGridSize,
    FaildedBackendInitialization,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error {}

#[derive(Debug, Clone, Copy)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
}

impl From<Color> for pixels::Color {
    fn from(c: Color) -> Self {
        pixels::Color::RGB(c.r, c.g, c.b)
    }
}

impl From<pixels::Color> for Color {
    fn from(c: pixels::Color) -> Self {
        Self {
            r: c.r,
            g: c.g,
            b: c.b,
        }
    }
}

impl From<(u8, u8, u8)> for Color {
    fn from(v: (u8, u8, u8)) -> Self {
        Self {
            r: v.0,
            g: v.1,
            b: v.2,
        }
    }
}

#[derive(Default)]
pub struct WindowConfig {
    pub title: String,
    pub w: u32,
    pub h: u32,
    pub vsync: bool,
    pub resizable: bool,
}

impl WindowConfig {
    pub fn new() -> WindowConfig {
        Self::default()
    }

    pub fn title(self, title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..self
        }
    }

    pub fn size(self, w: u32, h: u32) -> Self {
        Self { w, h, ..self }
    }

    pub fn vsync(self) -> Self {
        Self {
            vsync: true,
            ..self
        }
    }

    pub fn resizable(self) -> Self {
        Self {
            resizable: true,
            ..self
        }
    }

    pub fn verify(&self) -> Result<()> {
        if self.w == 0 || self.h == 0 {
            Err(Error::InvalidWindowSize)
        } else {
            Ok(())
        }
    }
}

struct Window {
    canvas: Canvas<video::Window>,
    w: u32,
    h: u32,
}

impl Window {
    pub fn set_size(&mut self, w: u32, h: u32) -> &mut Self {
        self.canvas.window_mut().set_size(w, h).unwrap();

        self.w = w;
        self.h = h;
        self
    }
}

#[derive(Default)]
pub struct GridConfig {
    pub w: u32,
    pub h: u32,
}

impl GridConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn size(self, w: u32, h: u32) -> Self {
        Self { w, h }
    }

    pub fn verify(&self) -> Result<()> {
        if self.w == 0 || self.h == 0 {
            Err(Error::InvalidGridSize)
        } else {
            Ok(())
        }
    }
}

pub struct Grid {
    buffer: Vec<u8>,
    texture: Texture,
    pitch: usize,
    dirty: bool,
    w: usize,
    h: usize,
}

pub struct Printer {
    _texture_creator: TextureCreator<WindowContext>,
    timer: Timer,
    window: Window,
    grid: Grid,
}

impl Printer {
    fn new(sdl_context: &Sdl, win_cfg: &WindowConfig, grid_cfg: &GridConfig) -> Result<Self> {
        let video_subsystem = sdl_context
            .video()
            .map_err(|_| Error::FaildedBackendInitialization)?;

        let mut window_builder = video_subsystem.window(&win_cfg.title, win_cfg.w, win_cfg.h);
        if win_cfg.resizable {
            window_builder.resizable();
        }
        let window = window_builder
            .build()
            .map_err(|_| Error::FaildedBackendInitialization)?;

        let mut canvas_builder = window.into_canvas().accelerated();
        if win_cfg.vsync {
            canvas_builder = canvas_builder.present_vsync();
        }
        let canvas = canvas_builder
            .build()
            .map_err(|_| Error::FaildedBackendInitialization)?;

        let texture_creator = canvas.texture_creator();
        let texture = texture_creator
            .create_texture_streaming(PixelFormatEnum::RGB24, grid_cfg.w, grid_cfg.h)
            .map_err(|_| Error::FaildedBackendInitialization)?;

        let grid = Grid {
            w: grid_cfg.w as usize,
            h: grid_cfg.h as usize,
            pitch: grid_cfg.w as usize * 3,
            dirty: true,
            buffer: vec![0; (grid_cfg.w * grid_cfg.h * 3) as usize],
            texture,
        };

        let window = Window {
            w: win_cfg.w,
            h: win_cfg.h,
            canvas,
        };

        let timer = Timer::new();

        let mut printer = Self {
            timer,
            grid,
            window,
            _texture_creator: texture_creator,
        };

        printer.set_clear_clr((0, 0, 0));

        Ok(printer)
    }

    pub fn buffer_mut(&mut self) -> &mut [u8] {
        self.grid.dirty = true;
        &mut self.grid.buffer
    }

    pub fn buffer(&self) -> &[u8] {
        &self.grid.buffer
    }

    pub fn pitch(&self) -> usize {
        self.grid.pitch
    }

    pub fn grid_size(&self) -> (usize, usize) {
        (self.grid.w, self.grid.h)
    }

    fn init_line(&self, x0: &mut f32, x1: &mut f32, y0: &mut f32, y1: &mut f32) -> f32 {
        if *y1 < *y0 {
            swap(y0, y1);
            swap(x0, x1);
        }
        let t = (*x1 - *x0) / (*y1 - *y0);
        if *y0 < 0.0 {
            *x0 -= *y0 * t;
            *y0 = 0.0;
        }
        let gw = self.grid.w as f32;
        if gw < *y1 {
            *x1 += (gw - *y0) * t;
            *y1 = gw;
        }

        t
    }

    pub fn line<C: Into<Color> + Copy, P: Into<Point>>(
        &mut self,
        p0: P,
        p1: P,
        color: C,
    ) -> &mut Self {
        let (mut p0, mut p1) = (p0.into(), p1.into());
        if (p0.x - p1.x).abs() < (p0.y - p1.y).abs() {
            let t = self.init_line(&mut p0.x, &mut p1.x, &mut p0.y, &mut p1.y);
            while p0.y < p1.y {
                self.pixel((p0.x, p0.y), color);
                p0.y += 1.0;
                p0.x += t;
            }
        } else {
            let t = self.init_line(&mut p0.y, &mut p1.y, &mut p0.x, &mut p1.x);
            while p0.x < p1.x {
                self.pixel((p0.x, p0.y), color);
                p0.x += 1.0;
                p0.y += t;
            }
        }
        self.pixel((p1.x, p1.y), color);

        self
    }

    fn normalize_point(&self, mut x: i32, mut y: i32) -> Option<(u32, u32)> {
        let (rx, ry, w, h) = self.calc_texture_rect();
        let (rx, ry, w, h) = (rx as i32, ry as i32, w as i32, h as i32);
        if x < rx || y < ry || rx + w < x || ry + h < y {
            None
        } else {
            x = (x - rx) * self.grid.w as i32 / w;
            y = (y - ry) * self.grid.h as i32 / h;
            Some((x as u32, y as u32))
        }
    }

    pub fn clear(&mut self) -> &mut Self {
        let clr = self.window.canvas.draw_color();
        self.clear_with_clr(clr)
    }

    pub fn clear_with_clr(&mut self, clr: impl Into<Color>) -> &mut Self {
        let pitch = self.pitch();
        let (w, h) = self.grid_size();
        let buff = self.buffer_mut();
        let clr = clr.into();
        for y in 0..h {
            for x in 0..w {
                let offset = x * BYTES_PER_PIXEL + y * pitch;
                buff[offset] = clr.r;
                buff[offset + 1] = clr.g;
                buff[offset + 2] = clr.b;
            }
        }
        self
    }

    pub fn clear_mono(&mut self, color: u8) -> &mut Self {
        self.grid.buffer.fill(color);
        self.grid.dirty = true;
        self
    }

    pub fn rect<C: Into<Color>, R: Into<Rect>>(&mut self, r: R, color: C) -> &mut Self {
        let Rect { x, y, w, h } = r.into();
        let (gw, gh) = (self.grid.w as f32, self.grid.h as f32);
        let x1 = x.clamp(0.0, gw - 1.0);
        let y1 = y.clamp(0.0, gh - 1.0);
        let x2 = (x1 + w).clamp(0.0, gw - 1.0);
        let y2 = (y1 + h).clamp(0.0, gh - 1.0);
        let clr = color.into();
        let pitch = self.pitch();
        let buffer = self.buffer_mut();

        for y in y1 as usize..=y2 as usize {
            for x in x1 as usize..=x2 as usize {
                let offset = (x * 3) + pitch * y;
                buffer[offset] = clr.r;
                buffer[offset + 1] = clr.g;
                buffer[offset + 2] = clr.b;
            }
        }

        self.grid.dirty = true;
        self
    }

    pub fn pixel<C: Into<Color>, P: Into<Point>>(&mut self, p: P, color: C) -> &mut Self {
        let Point { x, y } = p.into();
        let pitch = self.pitch();
        let grid = &mut self.grid;
        if grid.w as f32 <= x || grid.h as f32 <= y {
            return self;
        }
        let offset = (x as usize * 3) + pitch * y as usize;
        let c = color.into();
        grid.buffer[offset] = c.r;
        grid.buffer[offset + 1] = c.g;
        grid.buffer[offset + 2] = c.b;
        grid.dirty = true;
        self
    }

    pub fn set_clear_clr(&mut self, color: impl Into<Color>) -> &mut Self {
        self.window.canvas.set_draw_color(color.into());
        self
    }

    pub fn present(&mut self) {
        self.window.canvas.clear();

        if self.grid.dirty {
            self.grid
                .texture
                .update(None, &self.grid.buffer, self.grid.w * 3)
                .unwrap();
            self.grid.dirty = false;
        }

        let dst = self.calc_texture_sdl_rect();
        let (gw, gh) = (self.grid.w as u32, self.grid.h as u32);

        self.window
            .canvas
            .copy(&self.grid.texture, SDLRect::new(0, 0, gw, gh), dst)
            .unwrap();

        self.window
            .canvas
            .draw_rect(SDLRect::new(0, 0, 10, gh))
            .unwrap();

        self.window.canvas.present();

        self.timer.tick();
    }

    fn calc_texture_sdl_rect(&self) -> SDLRect {
        let (sw, sh) = self.window.canvas.window().drawable_size();
        let (gw, gh) = (self.grid.w as u32, self.grid.h as u32);

        let w;
        let h;

        if sw * gh < sh * gw {
            w = sw;
            h = gh * w / gw;
        } else {
            h = sh;
            w = gw * h / gh;
        }

        SDLRect::new((sw - w) as i32 / 2, (sh - h) as i32 / 2, w, h)
    }

    fn calc_texture_rect(&self) -> (u32, u32, u32, u32) {
        let (sw, sh) = self.window.canvas.window().drawable_size();
        let (gw, gh) = (self.grid.w as u32, self.grid.h as u32);

        let w;
        let h;

        if sw * gh < sh * gw {
            w = sw;
            h = gh * w / gw;
        } else {
            h = sh;
            w = gw * h / gh;
        }

        ((sw - w) / 2, (sh - h) / 2, w, h)
    }
}

#[derive(PartialEq, Eq)]
pub enum MouseButton {
    Unknown = 0,
    Left = 1,
    Middle = 2,
    Right = 3,
}

impl From<SDLMouseBtn> for MouseButton {
    fn from(value: SDLMouseBtn) -> Self {
        match value {
            SDLMouseBtn::Left => MouseButton::Left,
            SDLMouseBtn::Right => MouseButton::Right,
            SDLMouseBtn::Middle => MouseButton::Middle,
            _ => MouseButton::Unknown,
        }
    }
}

#[derive(PartialEq, Eq)]
pub enum MouseEvent {
    Up { btn: MouseButton, x: u32, y: u32 },
    Down { btn: MouseButton, x: u32, y: u32 },
    Motion { x: u32, y: u32 },
}

#[derive(PartialEq, Eq)]
pub enum Event {
    Quit,
    Mouse(MouseEvent),
    Unknown,
}

pub struct Timer {
    last_frame_duration: Duration,
    frame_start: Instant,
}

impl Timer {
    pub fn last_frame_duration(&self) -> Duration {
        self.last_frame_duration
    }

    fn tick(&mut self) {
        self.last_frame_duration = self.frame_start.elapsed();
        self.frame_start = Instant::now();
    }

    fn new() -> Self {
        Self {
            last_frame_duration: Duration::ZERO,
            frame_start: Instant::now(),
        }
    }
}

pub struct Keyboard {
    prev: HashSet<Keycode>,
    curr: HashSet<Keycode>,
}

impl Keyboard {
    fn new() -> Self {
        Self {
            prev: HashSet::new(),
            curr: HashSet::new(),
        }
    }

    fn update(&mut self, events: &EventPump) {
        swap(&mut self.curr, &mut self.prev);
        self.curr = events
            .keyboard_state()
            .pressed_scancodes()
            .filter_map(Keycode::from_scancode)
            .collect();
    }

    pub fn holded(&self, key: Keycode) -> bool {
        self.curr.contains(&key) && self.prev.contains(&key)
    }

    pub fn pressed(&self, key: Keycode) -> bool {
        self.curr.contains(&key) && !self.prev.contains(&key)
    }
}

pub struct Mouse {
    prev: HashSet<SDLMouseBtn>,
    curr: HashSet<SDLMouseBtn>,
    pub x: u32,
    pub y: u32,
}

impl Mouse {
    fn new() -> Self {
        Self {
            prev: HashSet::new(),
            curr: HashSet::new(),
            x: 0,
            y: 0,
        }
    }

    fn update(&mut self, events: &EventPump, printer: &Printer) {
        swap(&mut self.curr, &mut self.prev);
        let state = events.mouse_state();
        self.curr = state.pressed_mouse_buttons().collect();
        (self.x, self.y) = printer
            .normalize_point(state.x(), state.y())
            .unwrap_or((self.x, self.y));
    }

    pub fn holded(&self, btn: SDLMouseBtn) -> bool {
        self.curr.contains(&btn) && self.prev.contains(&btn)
    }

    pub fn pressed(&self, btn: SDLMouseBtn) -> bool {
        self.curr.contains(&btn) && !self.prev.contains(&btn)
    }
}

pub struct Pixer {
    printer: Printer,
    keyboard: Keyboard,
    mouse: Mouse,
    event_pump: EventPump,
    _sdl_context: Sdl,
}

impl Pixer {
    pub fn new(win_cfg: &WindowConfig, grid_cfg: &GridConfig) -> Result<Self> {
        win_cfg.verify()?;
        grid_cfg.verify()?;

        let sdl_context = sdl2::init().map_err(|_| Error::FaildedBackendInitialization)?;
        let printer = Printer::new(&sdl_context, win_cfg, grid_cfg)?;
        let event_pump = sdl_context
            .event_pump()
            .map_err(|_| Error::FaildedBackendInitialization)?;
        let keyboard = Keyboard::new();
        let mouse = Mouse::new();

        Ok(Pixer {
            keyboard,
            mouse,
            printer,
            _sdl_context: sdl_context,
            event_pump,
        })
    }

    pub fn timer(&self) -> &Timer {
        &self.printer.timer
    }

    pub fn timer_mut(&mut self) -> &mut Timer {
        &mut self.printer.timer
    }

    pub fn draw(&mut self) -> &mut Printer {
        &mut self.printer
    }

    fn handle_event(&mut self, event: SDLEvent) -> Option<Event> {
        match event {
            SDLEvent::Window { win_event, .. } => match win_event {
                WindowEvent::Resized(w, h) => {
                    self.printer.window.set_size(w as u32, h as u32);
                }
                _ => {}
            },
            SDLEvent::MouseButtonUp {
                mouse_btn, x, y, ..
            } => {
                if let Some((nx, ny)) = self.printer.normalize_point(x, y) {
                    return Some(Event::Mouse(MouseEvent::Up {
                        btn: mouse_btn.into(),
                        x: nx,
                        y: ny,
                    }));
                }
            }
            SDLEvent::MouseButtonDown {
                mouse_btn, x, y, ..
            } => {
                if let Some((nx, ny)) = self.printer.normalize_point(x, y) {
                    return Some(Event::Mouse(MouseEvent::Down {
                        btn: mouse_btn.into(),
                        x: nx,
                        y: ny,
                    }));
                }
            }
            SDLEvent::MouseMotion { x, y, .. } => {
                if let Some((nx, ny)) = self.printer.normalize_point(x, y) {
                    return Some(Event::Mouse(MouseEvent::Motion { x: nx, y: ny }));
                }
            }
            SDLEvent::Quit { .. }
            | SDLEvent::KeyDown {
                keycode: Some(Keycode::Escape),
                ..
            } => return Some(Event::Quit),
            _ => {}
        }

        Some(Event::Unknown)
    }

    pub fn poll_event(&mut self) -> Option<Event> {
        if let Some(event) = self.event_pump.poll_event() {
            self.handle_event(event)
        } else {
            None
        }
    }

    pub fn wait_event(&mut self) -> Event {
        loop {
            let event = self.event_pump.wait_event();
            if let Some(filtered) = self.handle_event(event) {
                return filtered;
            }
        }
    }

    pub fn update_keyboard_state(&mut self) {
        self.keyboard.update(&self.event_pump)
    }

    pub fn update_mouse_state(&mut self) {
        self.mouse.update(&self.event_pump, &self.printer)
    }

    pub fn keyboard(&self) -> &Keyboard {
        &self.keyboard
    }

    pub fn mouse(&self) -> &Mouse {
        &self.mouse
    }
}
