use std::ops::{Index, IndexMut};

pub trait Mesh {
    fn get_dx(&self) -> f64;

    fn get_cell(&self, i: usize) -> f64;

    fn get_length(&self) -> usize;

    fn get_num_physical_points(&self) -> usize;

    fn grid_points(&self) -> &[f64];

    fn ghost_left(&self) -> usize;

    fn ghost_right(&self) -> usize;
}

pub struct Mesh1d {
    dx: f64,
    num_points: usize,
    num_physical: usize,
    _num_ghost: usize,
    ghost_left: usize,
    ghost_right: usize,
    grid_points: Vec<f64>,
}

impl Mesh1d {
    pub fn new(x_start: f64, x_end: f64, num_points: usize, num_ghost: usize) -> Self {
        let dx = (x_end - x_start) / num_points as f64;
        let total_points = num_points + 2 * num_ghost;
        let grid_points: Vec<f64> = (0..total_points)
            .map(|i| {
                let i_physical = i as isize - num_ghost as isize;
                x_start + (i_physical as f64 + 0.5) * dx
            })
            .collect();
        Mesh1d {
            dx,
            num_points: total_points,
            num_physical: num_points,
            _num_ghost: 2 * num_ghost,
            ghost_left: num_ghost,
            ghost_right: num_ghost,
            grid_points,
        }
    }

    pub fn new_ppm_grid(
        x_start: f64,
        x_end: f64,
        num_points: usize,
        left_ghost: usize,
        right_ghost: usize,
    ) -> Self {
        let dx = (x_end - x_start) / num_points as f64;
        let total_points = num_points + left_ghost + right_ghost;
        let grid_points: Vec<f64> = (0..total_points)
            .map(|i| {
                let i_physical = (i as isize) - (left_ghost as isize);
                x_start + (i_physical as f64 + 0.5) * dx
            })
            .collect();
        Mesh1d {
            dx,
            num_points: total_points,
            num_physical: num_points,
            _num_ghost: left_ghost + right_ghost,
            ghost_left: left_ghost,
            ghost_right: right_ghost,
            grid_points,
        }
    }
}

impl Mesh for Mesh1d {
    fn get_dx(&self) -> f64 {
        self.dx
    }

    fn get_cell(&self, i: usize) -> f64 {
        self.grid_points[i]
    }

    fn get_length(&self) -> usize {
        self.num_points
    }

    fn get_num_physical_points(&self) -> usize {
        self.num_physical
    }

    fn grid_points(&self) -> &[f64] {
        &self.grid_points
    }

    fn ghost_left(&self) -> usize {
        self.ghost_left
    }

    fn ghost_right(&self) -> usize {
        self.ghost_right
    }
}

impl Index<usize> for Mesh1d {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.grid_points[index]
    }
}

impl IndexMut<usize> for Mesh1d {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.grid_points[index]
    }
}
