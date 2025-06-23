extern crate nalgebra as na;

//TRAITS
pub trait Reconstruction {
    fn reconstruction(&self, u: &na::DVector<f64>) -> (na::DVector<f64>, na::DVector<f64>);
}

pub trait Limiter {
    fn limit(&self, a: f64, b: f64) -> f64;
}

pub trait WENOReconstructionMode: Reconstruction {
    fn compute_modes(&self, u: &na::DVector<f64>) -> (na::DVector<f64>, na::DVector<f64>);
}

//STRUCTS
// Limiters
pub struct McLimiter;
pub struct MinmodLimiter;
pub struct VanLeerLimiter;

//tvd
pub struct TVDReconstruction<L: Limiter> {
    pub limiter: L,
}

//ppm
pub struct PPMReconstruction;

//weno
pub struct WENOReconstruction {
    pub gamma_l: f64,
    pub gamma_c: f64,
    pub gamma_r: f64,
    pub p: i32,
}

//HELPER FUNCTIONS
// tvd
fn tvd_reconstruction<L: Limiter>(
    u: &na::DVector<f64>,
    limiter: &L,
) -> (na::DVector<f64>, na::DVector<f64>) {
    let total = u.len();
    let n = total - 2;

    let mut vec_right: Vec<f64> = vec![0.0; total];
    let mut vec_left: Vec<f64> = vec![0.0; total];

    for i in 1..=n {
        let u_left = u[i] - u[i - 1];
        let u_right = u[i + 1] - u[i];

        let slope = limiter.limit(u_right, u_left);

        vec_left[i] = u[i] - 0.5 * slope;
        vec_right[i] = u[i] + 0.5 * slope;
    }

    vec_left[0] = vec_left[n];
    vec_right[0] = vec_right[n];

    vec_left[n + 1] = vec_left[1];
    vec_right[n + 1] = vec_right[1];

    (
        na::DVector::from_vec(vec_left),
        na::DVector::from_vec(vec_right),
    )
}

//ppm
fn ppm_reconstruction(u: &na::DVector<f64>) -> (na::DVector<f64>, na::DVector<f64>) {
    let total = u.len();
    let n = total - 3;

    let phys_right: Vec<f64> = (1..=n)
        .map(|i| {
            let delta_up1 = {
                let a = u[i + 2] - u[i - 1];
                let b = u[i + 1] - u[i];
                let min1 = (0.5 * (a + b)).abs().min(2.0 * a.abs());
                0.5 * (a.signum() + b.signum()) * (2.0 * b.abs().min(min1))
            };
            let delta_u = {
                let a = u[i + 1] - u[i];
                let b = u[i] - u[i - 1];
                let min1 = (0.5 * (a + b)).abs().min(2.0 * a.abs());
                0.5 * (a.signum() + b.signum()) * (2.0 * b.abs().min(min1))
            };
            u[i] + 0.5 * (u[i + 1] - u[i]) - (1.0 / 6.0) * (delta_up1 - delta_u)
        })
        .collect();

    let mut first_val = phys_right[0];
    let mut last_val = phys_right[n - 1];

    let mut right: Vec<f64> = std::iter::once(last_val)
        .chain(phys_right.into_iter())
        .chain(std::iter::once(first_val))
        .collect();

    let phys_left: Vec<f64> = right.iter().take(n).copied().collect();

    first_val = phys_left[0];
    last_val = phys_left[n - 1];

    let mut left: Vec<f64> = std::iter::once(last_val)
        .chain(phys_left.into_iter())
        .chain(std::iter::once(first_val))
        .collect();

    let (updated_phys_left, updated_phys_right): (Vec<f64>, Vec<f64>) = (1..=n)
        .map(|i| {
            let mut l_val = left[i];
            let mut r_val = right[i];
            let u_val = u[i];

            if (r_val - u_val) * (u_val - l_val) <= 0.0 {
                l_val = u_val;
                r_val = u_val;
            }

            if (r_val - l_val) * (u_val - 0.5 * (r_val + l_val)) > (r_val - l_val).powi(2) / 6.0 {
                l_val = 3.0 * u_val - 2.0 * r_val;
            }

            if -((r_val - l_val).powi(2)) / 6.0 > (r_val - l_val) * (u_val - 0.5 * (r_val + l_val))
            {
                r_val = 3.0 * u_val - 2.0 * l_val;
            }

            let u_x = r_val - l_val;
            let u_xx = 3.0 * r_val - 6.0 * u_val + 3.0 * l_val;

            let new_l = u_val - 0.5 * u_x + u_xx * (0.25 - 1.0 / 12.0);
            let new_r = u_val + 0.5 * u_x + u_xx * (0.25 - 1.0 / 12.0);

            (new_l, new_r)
        })
        .unzip();

    first_val = updated_phys_left[0];
    last_val = *updated_phys_left.last().unwrap();

    left = std::iter::once(last_val)
        .chain(updated_phys_left.into_iter())
        .chain(std::iter::once(first_val))
        .collect();

    first_val = updated_phys_right[0];
    last_val = *updated_phys_right.last().unwrap();

    right = std::iter::once(last_val)
        .chain(updated_phys_right.into_iter())
        .chain(std::iter::once(first_val))
        .collect();

    (na::DVector::from_vec(left), na::DVector::from_vec(right))
}

//weno
fn reconstruction_weno3(
    u: &na::DVector<f64>,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
) -> (na::DVector<f64>, na::DVector<f64>) {
    let length = u.len();
    let n = length - 4;
    let epsilon = 1e-12;

    let phys_left: Vec<f64> = (2..=(n + 1))
        .map(|i| {
            let u_lx = -2.0 * u[i - 1] + 0.5 * u[i - 2] + 1.5 * u[i];
            let u_lxx = 0.5 * u[i - 2] - u[i - 1] + 0.5 * u[i];

            let u_cx = 0.5 * (u[i + 1] - u[i - 1]);
            let u_cxx = 0.5 * u[i - 1] - u[i] + 0.5 * u[i + 1];

            let u_rx = -1.5 * u[i] + 2.0 * u[i + 1] - 0.5 * u[i + 2];
            let u_rxx = 0.5 * u[i] - u[i + 1] + 0.5 * u[i + 2];

            let is_l = u_lx.powi(2) + 13.0 / 3.0 * u_lxx.powi(2);
            let is_c = u_cx.powi(2) + 13. / 3.0 * u_cxx.powi(2);
            let is_r = u_rx.powi(2) + 13.0 / 3.0 * u_rxx.powi(2);

            let w_l_lin = gamma_l / (is_l + epsilon).powi(p);
            let w_c_lin = gamma_c / (is_c + epsilon).powi(p);
            let w_r_lin = gamma_r / (is_r + epsilon).powi(p);

            let w_l = w_l_lin / (w_l_lin + w_c_lin + w_r_lin);
            let w_c = w_c_lin / (w_l_lin + w_c_lin + w_r_lin);
            let w_r = w_r_lin / (w_l_lin + w_c_lin + w_r_lin);

            let u_x = w_l * u_lx + w_c * u_cx + w_r * u_rx;
            let u_xx = w_l * u_lxx + w_c * u_cxx + w_r * u_rxx;

            let u_val = u[i] - 0.5 * u_x + u_xx * (0.25 - 1.0 / 12.0);
            u_val
        })
        .collect();

    let phys_right: Vec<f64> = (2..=(n + 1))
        .map(|i| {
            let u_lx = -2.0 * u[i - 1] + 0.5 * u[i - 2] + 1.5 * u[i];
            let u_lxx = 0.5 * u[i - 2] - u[i - 1] + 0.5 * u[i];

            let u_cx = 0.5 * (u[i + 1] - u[i - 1]);
            let u_cxx = 0.5 * u[i - 1] - u[i] + 0.5 * u[i + 1];

            let u_rx = -1.5 * u[i] + 2.0 * u[i + 1] - 0.5 * u[i + 2];
            let u_rxx = 0.5 * u[i] - u[i + 1] + 0.5 * u[i + 2];

            let is_l = u_lx.powi(2) + 13.0 / 3.0 * u_lxx.powi(2);
            let is_c = u_cx.powi(2) + 13. / 3.0 * u_cxx.powi(2);
            let is_r = u_rx.powi(2) + 13.0 / 3.0 * u_rxx.powi(2);

            let w_l_lin = gamma_l / (is_l + epsilon).powi(p);
            let w_c_lin = gamma_c / (is_c + epsilon).powi(p);
            let w_r_lin = gamma_r / (is_r + epsilon).powi(p);

            let w_l = w_l_lin / (w_l_lin + w_c_lin + w_r_lin);
            let w_c = w_c_lin / (w_l_lin + w_c_lin + w_r_lin);
            let w_r = w_r_lin / (w_l_lin + w_c_lin + w_r_lin);

            let u_x = w_l * u_lx + w_c * u_cx + w_r * u_rx;
            let u_xx = w_l * u_lxx + w_c * u_cxx + w_r * u_rxx;

            let u_val = u[i] + 0.5 * u_x + u_xx * (0.25 - 1.0 / 12.0);
            u_val
        })
        .collect();

    let mut first_val = phys_left[0];
    let mut last_val = *phys_left.last().unwrap();

    let left: Vec<f64> = std::iter::once(last_val)
        .chain(phys_left.into_iter())
        .chain(std::iter::once(first_val))
        .collect();

    first_val = phys_right[0];
    last_val = *phys_right.last().unwrap();

    let right: Vec<f64> = std::iter::once(last_val)
        .chain(phys_right.into_iter())
        .chain(std::iter::once(first_val))
        .collect();

    (na::DVector::from_vec(left), na::DVector::from_vec(right))
}

fn modes_weno3(
    u: &na::DVector<f64>,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
) -> (na::DVector<f64>, na::DVector<f64>) {
    let length = u.len();
    let n = length - 4;
    let epsilon = 1e-12;

    let mut u_x = vec![0.0; n + 4];
    let mut u_xx = vec![0.0; n + 4];

    for i in 2..=(n + 1) {
        let u_lx = -2.0 * u[i - 1] + 0.5 * u[i - 2] + 1.5 * u[i];
        let u_lxx = 0.5 * u[i - 2] - u[i - 1] + 0.5 * u[i];

        let u_cx = 0.5 * (u[i + 1] - u[i - 1]);
        let u_cxx = 0.5 * u[i - 1] - u[i] + 0.5 * u[i + 1];

        let u_rx = -1.5 * u[i] + 2.0 * u[i + 1] - 0.5 * u[i + 2];
        let u_rxx = 0.5 * u[i] - u[i + 1] + 0.5 * u[i + 2];

        let is_l = u_lx.powi(2) + 13.0 / 3.0 * u_lxx.powi(2);
        let is_c = u_cx.powi(2) + 13. / 3.0 * u_cxx.powi(2);
        let is_r = u_rx.powi(2) + 13.0 / 3.0 * u_rxx.powi(2);

        let w_l_lin = gamma_l / (is_l + epsilon).powi(p);
        let w_c_lin = gamma_c / (is_c + epsilon).powi(p);
        let w_r_lin = gamma_r / (is_r + epsilon).powi(p);

        let w_l = w_l_lin / (w_l_lin + w_c_lin + w_r_lin);
        let w_c = w_c_lin / (w_l_lin + w_c_lin + w_r_lin);
        let w_r = w_r_lin / (w_l_lin + w_c_lin + w_r_lin);

        let tmp_x = w_l * u_lx + w_c * u_cx + w_r * u_rx;
        let tmp_xx = w_l * u_lxx + w_c * u_cxx + w_r * u_rxx;

        u_x[i] = tmp_x;
        u_xx[i] = tmp_xx;
    }

    (na::DVector::from_vec(u_x), na::DVector::from_vec(u_xx))
}

//IMPLEMENTATION
//tvd
impl Limiter for McLimiter {
    fn limit(&self, a: f64, b: f64) -> f64 {
        let min1 = (0.5 * (a + b)).abs().min(2.0 * a.abs());
        0.5 * (a.signum() + b.signum()) * (2.0 * b.abs()).min(min1)
    }
}

impl Limiter for MinmodLimiter {
    fn limit(&self, a: f64, b: f64) -> f64 {
        0.5 * (a.signum() + b.signum()) * a.abs().min(b.abs())
    }
}

impl Limiter for VanLeerLimiter {
    fn limit(&self, a: f64, b: f64) -> f64 {
        (a.signum() + b.signum()) * (a * b) / (a.abs() + b.abs())
    }
}

impl<L: Limiter> Reconstruction for TVDReconstruction<L> {
    fn reconstruction(
        &self,
        u: &nalgebra::DVector<f64>,
    ) -> (nalgebra::DVector<f64>, nalgebra::DVector<f64>) {
        tvd_reconstruction(u, &self.limiter)
    }
}

//ppm
impl Reconstruction for PPMReconstruction {
    fn reconstruction(
        &self,
        u: &nalgebra::DVector<f64>,
    ) -> (nalgebra::DVector<f64>, nalgebra::DVector<f64>) {
        ppm_reconstruction(u)
    }
}

//weno
impl Reconstruction for WENOReconstruction {
    fn reconstruction(
        &self,
        u: &nalgebra::DVector<f64>,
    ) -> (nalgebra::DVector<f64>, nalgebra::DVector<f64>) {
        reconstruction_weno3(u, self.gamma_l, self.gamma_c, self.gamma_r, self.p)
    }
}

impl WENOReconstructionMode for WENOReconstruction {
    fn compute_modes(
        &self,
        u: &nalgebra::DVector<f64>,
    ) -> (nalgebra::DVector<f64>, nalgebra::DVector<f64>) {
        modes_weno3(u, self.gamma_l, self.gamma_c, self.gamma_r, self.p)
    }
}
