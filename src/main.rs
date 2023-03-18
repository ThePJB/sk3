use glow::*;
use std::time::Instant;
use glutin::event::VirtualKeyCode;
use glutin::event::Event;
use glutin::event::WindowEvent;
use glutin::event::MouseButton;
use glutin::event::ElementState;
use cpal::traits::*;
use ringbuf::*;
use core::f32::consts::PI;

// put on github

// maybe we can kill 2 birds with one stone in terms of implementing try_move
// animate either try move or want move.
// apply will (update positions and) remove want move unless its on ice
// do move will turn want move into move


// MMMmmmk but why is there a gap where he stands still in a buffered movement sequence. isnt it interpolating the same time and then immediately checking and doing the next move
// or its an ice thing
// when on ice it will pause for SLIDE_T seconds between buffered movements

// elf can slide through real cool like leaving the title behind him (etched into the ice)
// lol imagine if i used the sprites for all kind of effects. could use them to make the checkered background
// could use them to do pokie wheels

// more and lazier snow,
// reset resets snow t
// add 'try move' and wall bump animation
// its a bit like the bell, maybe i didnt want feedback lol. but maybe you should try having precise inputs

// if entity is moving on snow play snow move noise
// if entity is sliding on ice continue playing ice slide noise.. hows that, high frequency popping/sizzling
// try subtle randomization of sound properties

// add more colour to the design
// what if we had flags like world 1... level 1... coming down the sides
// maybe a fancy border

// undo juicings / hold to undo more
// proper victory graphic
// other levels + level menu
// level transition

// needs a directed graph level

pub const TILE_SNOW: usize = 0;
pub const TILE_WALL: usize = 1;
pub const TILE_ICE: usize = 2;

pub const ENT_PLAYER: usize = 0;
pub const ENT_PRESENT: usize = 1;
pub const ENT_TARGET: usize = 2;
pub const ENT_CRATE: usize = 3;
pub const ENT_NONE: usize = 4;

pub const SLIDE_T: f64 = 0.11;
pub const VICTORY_T: f64 = 0.5;

pub struct CurrentLevel {
    w: usize,
    h: usize,
    tiles: Vec<usize>,
    static_ents: Vec<usize>,
    moving_ents: Vec<usize>,
    ent_move_x: Vec<i32>,
    ent_move_y: Vec<i32>,
    ent_dir_x: Vec<i32>,
}

pub struct HistoryEntry {
    moving_ents: Vec<usize>,
    ent_dir_x: Vec<i32>,
}

impl CurrentLevel {
    fn can_move(&self, i: usize, j: usize, dx: i32, dy: i32) -> bool {
        let ni = (i as i32 + dx) as usize;
        let nj = (j as i32 + dy) as usize;

        self.tiles[nj*self.w + ni] != TILE_WALL && (self.moving_ents[nj*self.w + ni] == ENT_NONE || self.can_move(ni, nj, dx, dy))
    }
    fn do_move(&mut self, i: usize, j: usize, dx: i32, dy: i32) {
        if !self.can_move(i, j, dx, dy) {
            self.ent_move_x[j*self.w + i] = 0;
            self.ent_move_y[j*self.w + i] = 0;
            return;
        }
        if self.moving_ents[j*self.w+i] == ENT_NONE { return; }

        let ni = (i as i32 + dx) as usize;
        let nj = (j as i32 + dy) as usize;

        if self.can_move(ni, nj, dx, dy) {
            self.do_move(ni, nj, dx, dy);
        }
        self.ent_move_x[j*self.w + i] = dx;
        if dx != 0 {
            self.ent_dir_x[j*self.w + i] = dx;
        }
        self.ent_move_y[j*self.w + i] = dy;
    }
    fn move_players(&mut self, dx: i32, dy: i32) {
        for i in 0..self.w {
            for j in 0..self.h {
                if self.moving_ents[j*self.w+i] == ENT_PLAYER {
                    if self.can_move(i, j, dx, dy) {
                        self.do_move(i, j, dx, dy);
                    }
                }
            }
        }
    }
    fn apply_move(&mut self) {
        let mut next_ents = vec![ENT_NONE; self.w*self.h];
        let mut next_move_x = vec![0; self.w*self.h];
        let mut next_move_y = vec![0; self.w*self.h];
        let mut next_dir_x = vec![0; self.w*self.h];
        for i in 0..self.w {
            for j in 0..self.h {
                if self.moving_ents[j*self.w+i] == ENT_NONE { continue; }
                let ni = (i as i32 + self.ent_move_x[j*self.w+i]) as usize;
                let nj = (j as i32 + self.ent_move_y[j*self.w+i]) as usize;
                next_ents[nj*self.w + ni] = self.moving_ents[j*self.w+i];
                next_dir_x[nj*self.w + ni] = self.ent_dir_x[j*self.w+i];
                if self.tiles[nj*self.w + ni] == TILE_ICE {
                    next_move_x[nj*self.w + ni] = self.ent_move_x[j*self.w + i];
                    next_move_y[nj*self.w + ni] = self.ent_move_y[j*self.w + i];
                }
            }
        }
        self.moving_ents = next_ents;
        self.ent_move_x = next_move_x;
        self.ent_move_y = next_move_y;
        self.ent_dir_x = next_dir_x;
    }
    fn can_move_any_player(&self, dx: i32, dy: i32) -> bool {
        for i in 0..self.w {
            for j in 0..self.h {
                if self.moving_ents[j*self.w + i] == ENT_PLAYER {
                    if self.can_move(i, j, dx, dy) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    fn should_do_again(&self) -> bool {
        for i in 0..self.w {
            for j in 0..self.h {
                let dx = self.ent_move_x[j*self.w + i];
                let dy = self.ent_move_y[j*self.w + i];
                if dx != 0 || dy != 0 {
                    return true;
                }
            }
        }
        return false;
    }
    fn do_again(&mut self) {
        for i in 0..self.w {
            for j in 0..self.h {
                if self.ent_move_x[j*self.w + i] != 0 || self.ent_move_y[j*self.w + i] != 0 {
                    if self.can_move(i, j, self.ent_move_x[j*self.w + i], self.ent_move_y[j*self.w + i]) {
                        self.do_move(i, j, self.ent_move_x[j*self.w + i], self.ent_move_y[j*self.w + i])
                    } else {
                        self.ent_move_x[j*self.w + i] = 0;
                        self.ent_move_y[j*self.w + i] = 0;
                    }
                }
            }
        }
    }
    fn remaining_presents(&self) -> usize {
        let mut acc = 0;
        for i in 0..self.w {
            for j in 0..self.h {
                if self.static_ents[j*self.w + i] == ENT_TARGET {
                    if self.moving_ents[j*self.w + i] != ENT_PRESENT {
                        acc += 1;
                    }
                }
            }
        }
        acc
    }
}

pub struct Level {
    title: String,
    tiles: Vec<usize>,
    entities: Vec<(usize, usize, usize)>,
    w: usize,
    h: usize,
}

impl Level {
    fn from(title: &str, contents: &str) -> Level {

        let w = contents.split("\n").map(|x| x.trim()).filter(|x| x.len() > 0).nth(0).unwrap().len();
        let h = contents.split("\n").map(|x| x.trim()).filter(|x| x.len() > 0).count();

        let mut tiles = vec![0usize; w*h];
        let mut entities = vec![];

        let mut i;
        let mut j = 0;

        for row in contents.split("\n").map(|x| x.trim()).filter(|x| x.len() > 0) {
            i = 0;
            if row.len() != w {
                dbg!(row.len(), w, row, i, j);
                panic!("bad level");
            }

            for char in row.chars() {
                match char {
                    ' ' => tiles[j*w+i] = TILE_SNOW,
                    '#' => tiles[j*w+i] = TILE_WALL,
                    '/' => tiles[j*w+i] = TILE_ICE,
                    'p' => {
                        tiles[j*w+i] = 0;
                        entities.push((i,j,ENT_PLAYER));
                    },
                    'P' => {
                        tiles[j*w+i] = 2;
                        entities.push((i,j,ENT_PLAYER));
                    },
                    'b' => {
                        tiles[j*w+i] = 0;
                        entities.push((i,j,ENT_PRESENT));
                    },
                    'B' => {
                        tiles[j*w+i] = 2;
                        entities.push((i,j,ENT_PRESENT));
                    },
                    'c' => {
                        tiles[j*w+i] = 0;
                        entities.push((i,j,ENT_CRATE));
                    },
                    'C' => {
                        tiles[j*w+i] = 2;
                        entities.push((i,j,ENT_CRATE));
                    },
                    't' => {
                        tiles[j*w+i] = 0;
                        entities.push((i,j,ENT_TARGET));
                    },
                    'T' => {
                        tiles[j*w+i] = 2;
                        entities.push((i,j,ENT_TARGET));
                    },
                    _ => {
                        panic!("forbidden");
                    }
                }

                i += 1;
            }

            j += 1;
        }

        let mut l = Level {
            title: title.to_string(),
            tiles,
            entities,
            w,
            h,
        };
        l
    }
    fn to_current(&self) -> CurrentLevel {
        let mut static_ents = vec![ENT_NONE; self.w*self.h];
        let mut moving_ents = vec![ENT_NONE; self.w*self.h];
        let ent_move_x = vec![0; self.w*self.h];
        let ent_move_y = vec![0; self.w*self.h];

        for idx in 0..self.entities.len() {
            let (i, j, e) = self.entities[idx];
            match e {
                ENT_PLAYER => moving_ents[j*self.w+i] = ENT_PLAYER,
                ENT_PRESENT => moving_ents[j*self.w+i] = ENT_PRESENT,
                ENT_CRATE => moving_ents[j*self.w+i] = ENT_CRATE,
                ENT_TARGET => static_ents[j*self.w+i] = ENT_TARGET,
                _ => panic!("unknown entity"),
            }
        }
        CurrentLevel {
            w: self.w, 
            h: self.h, 
            tiles: self.tiles.clone(), 
            static_ents, 
            moving_ents, 
            ent_move_x: ent_move_x.clone(), 
            ent_move_y,
            ent_dir_x: ent_move_x.clone(),
        }
    }
}


// extreme dopamine mode: scores queue up and get loaded in
// or if they replenish can you go infinite
// maybe with time constraint

// make the squares trippy fragment shader programs

// ====================
// Math
// ====================
// might be fucked since it was meant to be 32 bits
pub fn khash(mut state: usize) -> usize {
    state = (state ^ 2747636419).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state
}
pub fn krand(seed: usize) -> f32 {
    (khash(seed)&0x00000000FFFFFFFF) as f32 / 4294967295.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

#[derive(Clone, Copy, Debug)]
pub struct V2 {
    x: f32,
    y: f32,
}
fn v2(x: f32, y: f32) -> V2 { V2 { x, y } }
#[derive(Clone, Copy, Debug)]
pub struct V3 {
    x: f32,
    y: f32,
    z: f32,
}
fn v3(x: f32, y: f32, z: f32) -> V3 { V3 { x, y, z } }
#[derive(Clone, Copy, Debug)]
pub struct V4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}
fn v4(x: f32, y: f32, z: f32, w: f32) -> V4 { V4 { x, y, z, w } }

impl V2 {
    pub fn dot(&self, other: V2) -> f32 {
        self.x*other.x + self.y * other.y
    }
}
impl V3 {
    pub fn dot(&self, other: V3) -> f32 {
        self.x*other.x + self.y * other.y + self.z*other.z
    }
}
impl V4 {
    pub fn dot(&self, other: V4) -> f32 {
        self.x*other.x + self.y * other.y + self.z*other.z + self.w*other.w
    }
    pub fn tl(&self) -> V2 {v2(self.x, self.y)}
    pub fn br(&self) -> V2 {v2(self.x + self.z, self.y + self.w)}
    pub fn tr(&self) -> V2 {v2(self.x + self.z, self.y)}
    pub fn bl(&self) -> V2 {v2(self.x, self.y + self.w)}
    pub fn grid_child(&self, i: usize, j: usize, w: usize, h: usize) -> V4 {
        let cw = self.z / w as f32;
        let ch = self.w / h as f32;
        v4(self.x + cw * i as f32, self.y + ch * j as f32, cw, ch)
    }
    pub fn hsv_to_rgb(&self) -> V4 {
        let v = self.z;
        let hh = (self.x % 360.0) / 60.0;
        let i = hh.floor() as i32;
        let ff = hh - i as f32;
        let p = self.z * (1.0 - self.y);
        let q = self.z * (1.0 - self.y * ff);
        let t = self.z * (1.0 - self.y * (1.0 - ff));
        match i {
            0 => v4(v, t, p, self.w),
            1 => v4(q, v, p, self.w),
            2 => v4(p, v, t, self.w),
            3 => v4(p, q, v, self.w),
            4 => v4(t, p, v, self.w),
            5 => v4(v, p, q, self.w),
            _ => panic!("unreachable"),
        }
    }
    fn contains(&self, p: V2) -> bool {
        !(p.x < self.x || p.x > self.x + self.z || p.y < self.y || p.y > self.y + self.w)
    }
    fn point_within(&self, p: V2) -> V2 {
        v2(p.x*self.z+self.x, p.y*self.w+self.y)
    }
    fn point_without(&self, p: V2) -> V2 {
        v2((p.x - self.x) / self.z, (p.y - self.y) / self.w)
    }
    fn fit_aspect(&self, a: f32) -> V4 {
        let a_self = self.z/self.w;

        if a_self > a {
            // parent wider
            v4((self.z - self.z*(1.0/a))/2.0, 0.0, self.z*1.0/a, self.w)
        } else {
            // child wider
            v4(0.0, (self.w - self.w*(1.0/a))/2.0, self.z, self.w*a)
        }
    }
}


// ====================
// Canvas
// ====================
pub struct CTCanvas {
    buf: Vec<u8>,
}

impl CTCanvas {
    pub fn new() -> CTCanvas {
        CTCanvas {
            buf: Vec::new(),
        }
    }

    fn put_u32(&mut self, x: u32) {
        for b in x.to_le_bytes() {
            self.buf.push(b);
        }
    }

    fn put_float(&mut self, x: f32) {
        for b in x.to_le_bytes() {
            self.buf.push(b);
        }
    }

    pub fn put_vertex(&mut self, p: V3, uv: V2, col: V4, mode: u32) {
        self.put_float(p.x);
        self.put_float(p.y);
        self.put_float(p.z);
        self.put_float(col.x);
        self.put_float(col.y);
        self.put_float(col.z);
        self.put_float(col.w);
        self.put_float(uv.x);
        self.put_float(uv.y);
        self.put_u32(mode);
    }
    pub fn put_triangle(&mut self, p1: V2, uv1: V2, p2: V2, uv2: V2, p3: V2, uv3: V2, depth: f32, colour: V4, mode: u32) {
        self.put_vertex(v3(p1.x, p1.y, depth), uv1, colour, mode);
        self.put_vertex(v3(p2.x, p2.y, depth), uv2, colour, mode);
        self.put_vertex(v3(p3.x, p3.y, depth), uv3, colour, mode);
    }
    pub fn put_quad(&mut self, p1: V2, uv1: V2, p2: V2, uv2: V2, p3: V2, uv3: V2, p4: V2, uv4: V2, depth: f32, colour: V4, mode: u32) {
        self.put_triangle(p1, uv1, p2, uv2, p3, uv3, depth, colour, mode);
        self.put_triangle(p4, uv4, p2, uv2, p3, uv3, depth, colour, mode);
    }

    pub fn put_rect(&mut self, r: V4, r_uv: V4, depth: f32, colour: V4, mode: u32) {
        self.put_triangle(r.tl(), r_uv.tl(), r.tr(), r_uv.tr(), r.bl(), r_uv.bl(), depth, colour, mode);
        self.put_triangle(r.bl(), r_uv.bl(), r.tr(), r_uv.tr(), r.br(), r_uv.br(), depth, colour, mode);
    }

    pub fn put_rect_flipx(&mut self, r: V4, r_uv: V4, depth: f32, colour: V4, mode: u32) {
        self.put_triangle(r.tl(), r_uv.tr(), r.tr(), r_uv.tl(), r.bl(), r_uv.br(), depth, colour, mode);
        self.put_triangle(r.bl(), r_uv.br(), r.tr(), r_uv.tl(), r.br(), r_uv.bl(), depth, colour, mode);
    }

    pub fn put_glyph(&mut self, c: char, r: V4, depth: f32, colour: V4) {
        let clip_fn = |mut c: u8| {
            if c >= 'a' as u8 && c <= 'z' as u8 {
                c -= 'a' as u8 - 'A' as u8;
            }
            if c >= '+' as u8 && c <= '_' as u8 {
                let x = c - '+' as u8;
                let w = '_' as u8 - '+' as u8 + 1; // maybe +1
                Some(v4(0.0, 0.0, 1.0, 0.5).grid_child(x as usize, 0, w as usize, 1))
            } else {
                None
            }
        };
        if let Some(r_uv) = clip_fn(c as u8) {
            self.put_rect(r, r_uv, depth, colour, 1);
        }
    }

    pub fn put_sprite(&mut self, idx: usize, r: V4, depth: f32, colour: V4) {
        let r_uv = v4(0.0, 0.5, 40.0/39.75, 0.5).grid_child(idx as usize, 0, 40 as usize, 1);
        self.put_rect(r, r_uv, depth, colour, 1);
    }

    pub fn put_sprite_flipx(&mut self, idx: usize, r: V4, depth: f32, colour: V4) {
        let r_uv = v4(0.0, 0.5, 40.0/39.75, 0.5).grid_child(idx as usize, 0, 40 as usize, 1);
        self.put_rect_flipx(r, r_uv, depth, colour, 1);
    }

    pub fn put_string_left(&mut self, s: &str, mut x: f32, y: f32, cw: f32, ch: f32, depth: f32, colour: V4) {
        for c in s.chars() {
            self.put_glyph(c, v4(x, y, cw, ch), depth, colour);
            x += cw;
        }
    }
    pub fn put_string_centered(&mut self, s: &str, mut x: f32, mut y: f32, cw: f32, ch: f32, depth: f32, colour: V4) {
        let w = s.len() as f32 * cw;
        x -= w/2.0;
        // y -= ch/2.0;
        for c in s.chars() {
            self.put_glyph(c, v4(x, y, cw, ch), depth, colour);
            x += cw;
        }
    }
}

// ====================
// Audio stuff
// ====================
// 0 : kick drum
// 1 : sad ding

fn sample_next(o: &mut SampleRequestOptions) -> f32 {
    let mut acc = 0.0;
    let mut idx = o.sounds.len();
    loop {
        if idx == 0 {
            break;
        }
        idx -= 1;

        if o.sounds[idx].wait > 0.0 {
            o.sounds[idx].wait -= 1.0/44100.0;
            continue;
        }

        o.sounds[idx].elapsed += 1.0/44100.0;
        o.sounds[idx].remaining -= 1.0/44100.0;

        let t = o.sounds[idx].elapsed;

        if o.sounds[idx].remaining < 0.0 {
            o.sounds.swap_remove(idx);
            continue;
        }
        if o.sounds[idx].id == 0 {
            o.sounds[idx].magnitude *= 0.999;

            let f = o.sounds[idx].frequency;
            let f_trans = f*3.0;

            let t_trans = 1.0/(2.0*PI*f_trans);

            if o.sounds[idx].elapsed < t_trans {
                o.sounds[idx].phase += f_trans*2.0*PI*1.0/o.sample_rate;
            } else {
                o.sounds[idx].phase += f*2.0*PI*1.0/o.sample_rate;
            }
            // o.sounds[idx].phase += f*2.0*PI*1.0/o.sample_rate;

            //o.sounds[idx].phase = o.sounds[idx].phase % 2.0*PI; // this sounds really good lol

            acc += (o.sounds[idx].phase).sin() * o.sounds[idx].magnitude
        } else if o.sounds[idx].id == 1 {
            o.sounds[idx].magnitude *= o.sounds[idx].mag_exp;
            o.sounds[idx].frequency *= o.sounds[idx].freq_exp;
            o.sounds[idx].phase += o.sounds[idx].frequency*2.0*PI*1.0/o.sample_rate;
            acc += (o.sounds[idx].phase).sin() * o.sounds[idx].magnitude
        }
    }
    acc
}

#[derive(Debug)]
pub struct Sound {
    id: usize,
    wait: f32,
    birthtime: f32,
    elapsed: f32,
    remaining: f32,
    magnitude: f32,
    mag_exp: f32,
    frequency: f32,
    freq_exp: f32,
    phase: f32,
}

pub struct SampleRequestOptions {
    pub sample_rate: f32,
    pub nchannels: usize,
    pub channel: Consumer<Sound>,
    pub sounds: Vec<Sound>,
}

pub fn stream_setup_for<F>(on_sample: F, channel: Consumer<Sound>) -> Result<cpal::Stream, anyhow::Error>
where
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let (_host, device, config) = host_device_setup()?;

    match config.sample_format() {
        cpal::SampleFormat::F32 => stream_make::<f32, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::I16 => stream_make::<i16, _>(&device, &config.into(), on_sample, channel),
        cpal::SampleFormat::U16 => stream_make::<u16, _>(&device, &config.into(), on_sample, channel),
    }
}

pub fn host_device_setup(
) -> Result<(cpal::Host, cpal::Device, cpal::SupportedStreamConfig), anyhow::Error> {
    let host = cpal::default_host();

    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow::Error::msg("Default output device is not available"))?;
    println!("Output device : {}", device.name()?);

    let config = device.default_output_config()?;
    println!("Default output config : {:?}", config);

    Ok((host, device, config))
}


pub fn stream_make<T, F>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    on_sample: F,
    channel: Consumer<Sound>,
) -> Result<cpal::Stream, anyhow::Error>
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static + Copy,
{
    let sample_rate = config.sample_rate.0 as f32;
    let nchannels = config.channels as usize;
    let mut request = SampleRequestOptions {
        sample_rate,
        nchannels,
        sounds: vec![],
        channel,
    };
    let err_fn = |err| eprintln!("Error building output sound stream: {}", err);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
            on_window(output, &mut request, on_sample)
        },
        err_fn,
    )?;

    Ok(stream)
}

fn on_window<T, F>(output: &mut [T], request: &mut SampleRequestOptions, mut on_sample: F)
where
    T: cpal::Sample,
    F: FnMut(&mut SampleRequestOptions) -> f32 + std::marker::Send + 'static,
{
    if let Some(sc) = request.channel.pop() {
        request.sounds.push(sc);
    }
    for frame in output.chunks_mut(request.nchannels) {
        let value: T = cpal::Sample::from::<f32>(&on_sample(request));
        for sample in frame.iter_mut() {
            *sample = value;
        }
    }
}


fn main() {
    unsafe {
        let mut xres = 1600i32;
        let mut yres = 1600i32;

        // ====================
        // Sound Init
        // ====================
        let rb = RingBuffer::<Sound>::new(100);
        let (mut prod, mut cons) = rb.split();
        let stream = stream_setup_for(sample_next, cons).expect("no can make stream");
        stream.play().expect("no can play stream");

        let event_loop = glutin::event_loop::EventLoop::new();
        let window_builder = glutin::window::WindowBuilder::new()
                .with_title("Snowkoban")
                .with_inner_size(glutin::dpi::PhysicalSize::new(xres, yres));

        let window = glutin::ContextBuilder::new()
                .with_vsync(true)
                .build_windowed(window_builder, &event_loop)
                .unwrap()
                .make_current()
                .unwrap();


        // ====================
        // GL init
        // ====================
        let gl = glow::Context::from_loader_function(|s| window.get_proc_address(s) as *const _);
        gl.enable(DEPTH_TEST);
        // gl.enable(CULL_FACE);
        gl.blend_func(SRC_ALPHA, ONE_MINUS_SRC_ALPHA);
        gl.enable(BLEND);
        // gl.debug_message_callback(|a, b, c, d, msg| {
        //     println!("{} {} {} {} msg: {}", a, b, c, d, msg);
        // });

        let vbo = gl.create_buffer().unwrap();
        gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));

        let vao = gl.create_vertex_array().unwrap();
        gl.bind_vertex_array(Some(vao));
        
        gl.vertex_attrib_pointer_f32(0, 3, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 0);
        gl.enable_vertex_attrib_array(0);
        gl.vertex_attrib_pointer_f32(1, 4, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 4*3);
        gl.enable_vertex_attrib_array(1);
        gl.vertex_attrib_pointer_f32(2, 2, glow::FLOAT, false, 4*4 + 4*3 + 4*2 + 4, 4*3 + 4*4);
        gl.enable_vertex_attrib_array(2);
        gl.vertex_attrib_pointer_i32(3, 1, glow::UNSIGNED_INT, 4*4 + 4*3 + 4*2 + 4, 4*3 + 4*4 + 4*2);
        gl.enable_vertex_attrib_array(3);


        // Shader
        let program = gl.create_program().expect("Cannot create program");
    
        let vs = gl.create_shader(glow::VERTEX_SHADER).expect("cannot create vertex shader");
        gl.shader_source(vs, include_str!("shader.vert"));
        gl.compile_shader(vs);
        if !gl.get_shader_compile_status(vs) {
            panic!("{}", gl.get_shader_info_log(vs));
        }
        gl.attach_shader(program, vs);

        let fs = gl.create_shader(glow::FRAGMENT_SHADER).expect("cannot create fragment shader");
        gl.shader_source(fs, include_str!("shader.frag"));
        gl.compile_shader(fs);
        if !gl.get_shader_compile_status(fs) {
            panic!("{}", gl.get_shader_info_log(fs));
        }
        gl.attach_shader(program, fs);

        gl.link_program(program);
        if !gl.get_program_link_status(program) {
            panic!("{}", gl.get_program_info_log(program));
        }
        gl.detach_shader(program, fs);
        gl.delete_shader(fs);
        gl.detach_shader(program, vs);
        gl.delete_shader(vs);

        let png_bytes = include_bytes!("../tex.png").as_ref();
        let decoder = png::Decoder::new(png_bytes);
        let mut reader = decoder.read_info().unwrap();
        let mut buf = vec![0; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        let bytes = &buf[..info.buffer_size()];

        let texture = gl.create_texture().unwrap();
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        gl.tex_image_2d(
            glow::TEXTURE_2D, 
            0, 
            glow::RGBA as i32, 
            info.width as i32, info.height as i32, 
            0, 
            RGBA, 
            glow::UNSIGNED_BYTE, 
            Some(bytes)
        );
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, glow::NEAREST as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE as i32);
        gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE as i32);
        gl.generate_mipmap(glow::TEXTURE_2D);





        // ====================
        // Simulation
        // ====================
        let worlds: Vec<(&str, Vec<Level>)> = vec![
            ("world1",vec![
                Level::from("ice to meet you", include_str!("levels/world2/1.lvl")),
                Level::from("wasd to move", include_str!("levels/world1/0.lvl")),
                Level::from("z to undo", include_str!("levels/world1/1.lvl")),
                Level::from("failure is the key to success", include_str!("levels/world1/2.lvl")),
                Level::from("greed test", include_str!("levels/world1/4.lvl")),
                Level::from("get around it", include_str!("levels/world1/3.lvl")),
            ]),
            ("world2",vec![
                Level::from("ice to meet you", include_str!("levels/world2/1.lvl")),
                Level::from("blocking", include_str!("levels/world2/2.lvl")),
                Level::from("dont get stuck on this one", include_str!("levels/world2/3.lvl")),
            ]),
        ];

        let mut curr_world_num = 0;
        let mut curr_level_num = 0;

        let mut t = 0.0;
        let mut t_move = 0.0;
        let mut t_last = Instant::now();
        let mut mouse_pos = v2(0., 0.);

        let mut animating = false;
        let mut victory = false;
        let mut buffered_dx = 0;
        let mut buffered_dy = 0;

        let mut curr_level = worlds[curr_world_num].1[curr_level_num].to_current();

        let mut history: Vec<HistoryEntry> = Vec::new();

        let mut menu = false;

        event_loop.run(move |event, _, _| {

            match event {
                Event::LoopDestroyed |
                Event::WindowEvent {event: WindowEvent::CloseRequested, ..} => {
                    std::process::exit(0);
                }

                Event::WindowEvent {event, .. } => {
                    match event {
                        WindowEvent::CursorMoved {position, .. } => {
                            mouse_pos.x = position.x as f32 / xres as f32;
                            mouse_pos.y = position.y as f32 / yres as f32;
                        },
                        WindowEvent::Resized(size) => {
                            xres = size.width as i32;
                            yres = size.height as i32;
                            gl.viewport(0, 0, size.width as i32, size.height as i32)
                        },
                        WindowEvent::MouseInput {state: ElementState::Pressed, button: MouseButton::Left, ..} => {
                        },
                        WindowEvent::KeyboardInput {input, ..} => {
                            match input {
                                glutin::event::KeyboardInput {virtual_keycode: Some(code), state: ElementState::Released, ..} => {
                                    if victory && t - t_move > VICTORY_T {
                                        curr_level_num += 1;
                                        if curr_level_num == worlds[curr_world_num].1.len() {
                                            curr_level_num = 0;
                                            if curr_world_num != worlds.len() - 1 {
                                                curr_world_num += 1;
                                            }
                                            menu = true;
                                        }
                                        dbg!(curr_world_num, curr_level_num);
                                        curr_level = worlds[curr_world_num].1[curr_level_num].to_current();
                                        victory = false;
                                        buffered_dx = 0;
                                        buffered_dy = 0;
                                        animating = false;
                                        history = vec![];
                                        return;
                                    }
                                    match code {
                                        VirtualKeyCode::Escape => {
                                        },
                                        VirtualKeyCode::M => {
                                            victory = true
                                        },
                                        VirtualKeyCode::W | VirtualKeyCode::Up |
                                        VirtualKeyCode::S | VirtualKeyCode::Down |
                                        VirtualKeyCode::A | VirtualKeyCode::Left |
                                        VirtualKeyCode::D | VirtualKeyCode::Right => {
                                            let (dx, dy) = match code {
                                                VirtualKeyCode::W | VirtualKeyCode::Up => (0, -1),
                                                VirtualKeyCode::S | VirtualKeyCode::Down => (0, 1),
                                                VirtualKeyCode::A | VirtualKeyCode::Left => (-1, 0),
                                                VirtualKeyCode::D | VirtualKeyCode::Right =>(1, 0),
                                                _ => panic!("unreachable"),
                                            };
                                            if animating {
                                                buffered_dx = dx;
                                                buffered_dy = dy;
                                            } else if !victory {
                                                if curr_level.can_move_any_player(dx, dy) {
                                                    animating = true;
                                                    history.push(HistoryEntry {
                                                        moving_ents: curr_level.moving_ents.clone(),
                                                        ent_dir_x: curr_level.ent_dir_x.clone(),
                                                    });
                                                    curr_level.move_players(dx, dy);
                                                    t_move = t;
                                                    prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 440.0, freq_exp: 1.0, wait: 0.0, phase: 0.0 }).unwrap();
                                                } else {
                                                    prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 220.0, freq_exp: 0.999, wait: 0.0, phase: 0.0 }).unwrap();
                                                }
                                            }
                                        },

                                        VirtualKeyCode::Z => {
                                            if !victory && !animating {
                                                if let Some(he) = history.pop() {
                                                    curr_level.moving_ents = he.moving_ents;
                                                    curr_level.ent_dir_x = he.ent_dir_x;
                                                    prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.2, magnitude: 0.1, mag_exp: 0.9999, frequency: 100.0, freq_exp: 1.0002, wait: 0.0, phase: 0.0 }).unwrap();
                                                }
                                            }
                                        },
                                        _ => {},
                                    }
                                },
                                _ => {},
                            }
                        },
                        _ => {},
                    }
                }

                Event::MainEventsCleared => {
                    let t_now = Instant::now();
                    let dt = (t_now - t_last).as_secs_f64();
                    t += dt;
                    t_last = t_now;

                    let mut canvas = CTCanvas::new();

                    let w = curr_level.w;
                    let h = curr_level.h;

                    let screen_rect = v4(-1., -1., 2., 2.);
                    let aspect = xres as f32 / yres as f32;
                    let level_aspect = w as f32 / h as f32;

                    let lvlw = (2.0*level_aspect/aspect).min(2.0);
                    let lvlh = (2.0*aspect/level_aspect).min(2.0);
                    let xo = (2.0 - lvlw)/2.0;
                    let yo = (2.0 - lvlh)/2.0;
                    let lvlx = -1. + xo;
                    let lvly = -1. + yo;

                    let level_rect = v4(lvlx, lvly, lvlw, lvlh);

                    if t - t_move > SLIDE_T {
                        let n_pres = curr_level.remaining_presents();
                        curr_level.apply_move();
                        let n_pres2 = curr_level.remaining_presents();
                        if n_pres2 < n_pres {
                            prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 440.0 * 3./2., freq_exp: 1.0, wait: 0.0, phase: 0.0 }).unwrap();
                            if n_pres2 == 0 {
                                prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 440.0 * 2., freq_exp: 1.0, wait: SLIDE_T as f32, phase: 0.0 }).unwrap();
                            }
                        }
                        if n_pres2 > n_pres {
                            prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 440.0 / (3./2.), freq_exp: 1.0, wait: 0.0, phase: 0.0 }).unwrap();
                        }
                        if curr_level.should_do_again() {
                            t_move = t;
                            curr_level.do_again();
                        } else {
                            if buffered_dx != 0 || buffered_dy != 0 && !victory {
                                if curr_level.can_move_any_player(buffered_dx, buffered_dy) {
                                    animating = true;
                                    t_move = t;
                                    history.push(HistoryEntry {
                                        moving_ents: curr_level.moving_ents.clone(),
                                        ent_dir_x: curr_level.ent_dir_x.clone(),
                                    });
                                    curr_level.move_players(buffered_dx, buffered_dy);

                                    prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 440.0, freq_exp: 1.0, wait: 0.0, phase: 0.0 }).unwrap();
                                } else {
                                    prod.push(Sound { id: 1, birthtime: t as f32, elapsed: 0.0, remaining: 0.1, magnitude: 0.1, mag_exp: 0.999, frequency: 220.0, freq_exp: 0.999, wait: 0.0, phase: 0.0 }).unwrap();
                                }
                                buffered_dx = 0;
                                buffered_dy = 0;
                            } else {
                                animating = false;
                            }
                        }
                    }



                    // sliding: for undos whole slides need to be accounted for
                    // but whether something loses its aspiration to move

                    if curr_level.remaining_presents() == 0 &&!victory {
                        victory = true;
                        animating = true;
                    }


                    // canvas.put_rect(screen_rect, v4(0., 0., 1., 1.), 0.9, v4(0.2, 0.2, 0.2, 1.0), 0);
                    canvas.put_rect(screen_rect, v4(0., 0., 1., 1.), 0.9, v4(0.2, 0.2, 0.2, 1.0), 0);
                    canvas.put_rect(level_rect, v4(0., 0., 1., 1.), 0.0, v4(0.7, 0.2, 0.6, 1.0), 0);

                    // computation of UVs for drawing of background out of wall sprite
                    let tw = lvlw/w as f32;
                    let th = lvlh/h as f32;
                    let nw = (screen_rect.z / tw).ceil() + 1.;
                    let nh = (screen_rect.w / th).ceil() + 1.;

                    let xstart = xo - (xo/tw).ceil()*tw - 1.;
                    let ystart = yo - (yo/th).ceil()*th - 1.;

                    // background wall texture
                    for i in 0..nw as usize {
                        for j in 0..nh as usize {
                            let x = xstart + i as f32 * tw;
                            let y = ystart + j as f32 * th;
                            canvas.put_sprite(0, v4(x, y, tw, th), 0.8, v4(1., 1., 1., 1.));
                        }
                    }

                    // tiles
                    for i in 0..w {
                        for j in 0..h {
                            let sprite = match curr_level.tiles[j*w + i] {
                                TILE_WALL => 0,
                                TILE_SNOW => 1,
                                TILE_ICE => 2,
                                _ => continue,
                            };

                            let r = level_rect.grid_child(i, j, w, h);

                            canvas.put_sprite(sprite, r, -0.1, v4(1., 1., 1., 1.));
                        }
                    }

                    // entities
                    for i in 0..curr_level.w {
                        for j in 0..curr_level.h {
                            if curr_level.static_ents[j*curr_level.w+i] != ENT_TARGET { continue; }
                            let r = level_rect.grid_child(i, j, w, h);
                            canvas.put_sprite(6, r, -0.19, v4(1., 1., 1., 1.));
                        }
                    }

                    for i in 0..curr_level.w {
                        for j in 0..curr_level.h {
                            let sprite = match curr_level.moving_ents[j*curr_level.w+i] {
                                ENT_PLAYER => if t % 1.0 > 0.5 {
                                    4 
                                } else {
                                    5
                                },
                                ENT_CRATE => 7,
                                ENT_PRESENT => 9,
                                _ => continue,
                            };
                            let mut r = level_rect.grid_child(i, j, w, h);
                            let x_shift = curr_level.ent_move_x[j*curr_level.w+i] as f32 * r.w * ((t - t_move)/SLIDE_T) as f32;
                            let y_shift = curr_level.ent_move_y[j*curr_level.w+i] as f32 * r.z * ((t - t_move)/SLIDE_T) as f32;
                            r.x += x_shift;
                            r.y += y_shift;

                            let colour = v4(1., 1., 1., 1.);

                            if curr_level.moving_ents[j*curr_level.w+i] == ENT_PLAYER && curr_level.ent_dir_x[j*curr_level.w+i] >= 0 {
                                canvas.put_sprite_flipx(sprite, r, -0.2, colour);
                            } else {
                                canvas.put_sprite(sprite, r, -0.2, colour);
                            };
                        }
                    }

                    // ok how we drawin that snow
                    // gotta transform it to be not stretched also
                    let num_cols = 60.0 * aspect;
                    for i in 0..num_cols.floor() as usize + 10 {
                        let col_x = 2.0 * i as f32 / num_cols as f32;
                        let col_seed = khash(i + 12312947);
                        let phase = (krand(col_seed) + 0.05*t as f32) % 1.0;
                        let y = phase * 2.5 - 1.5;
                        let x = (col_x + phase * 0.05 + 0.07*(10.0 * phase + 2.0*PI*krand(col_seed+1238124517)).sin()) * 2.0 - 1.5;
                        let r = v4(x, y, 0.03/aspect, 0.03);
                        canvas.put_rect(r.grid_child(1, 0, 3, 3), v4(0., 0., 1., 1.), -0.3, v4(1., 1., 1., 1.), 0);
                        canvas.put_rect(r.grid_child(1, 2, 3, 3), v4(0., 0., 1., 1.), -0.3, v4(1., 1., 1., 1.), 0);
                        canvas.put_rect(r.grid_child(0, 1, 3, 3), v4(0., 0., 1., 1.), -0.3, v4(1., 1., 1., 1.), 0);
                        canvas.put_rect(r.grid_child(2, 1, 3, 3), v4(0., 0., 1., 1.), -0.3, v4(1., 1., 1., 1.), 0);
                    }

                    // print level name
                    canvas.put_string_left(&worlds[curr_world_num].1[curr_level_num].title, -1., 1. - 0.06, 0.06*14./16.*yres as f32/xres as f32, 0.06, -0.4, v4(1., 1., 1., 1.));

                    // string may need to be * aspect

                    // demonstrate victory
                    // unfurling banner where victory fades in
                    if victory {
                        canvas.put_string_centered("victory", 0.0, 0.0, 0.1*14./16.*yres as f32/xres as f32, 0.1, -0.4, v4(1., 1., 1., 1.,));
                    }

                    gl.uniform_1_f32(gl.get_uniform_location(program, "time").as_ref(), t as f32);

                    gl.clear_color(0.5, 0.5, 0.5, 1.0);
                    gl.clear(glow::COLOR_BUFFER_BIT | glow::DEPTH_BUFFER_BIT); 
                    gl.bind_texture(glow::TEXTURE_2D, Some(texture));
                    gl.use_program(Some(program));
                    gl.bind_vertex_array(Some(vao));
                    gl.bind_buffer(glow::ARRAY_BUFFER, Some(vbo));
                    gl.buffer_data_u8_slice(glow::ARRAY_BUFFER, &canvas.buf, glow::DYNAMIC_DRAW);
                    let vert_count = canvas.buf.len() / (10*4);
                    gl.draw_arrays(glow::TRIANGLES, 0, vert_count as i32);
                    window.swap_buffers().unwrap();
                }

                _ => {},
            }
        });

    }
}