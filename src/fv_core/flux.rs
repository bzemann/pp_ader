extern crate nalgebra as na;

pub trait AnaFluxADERSpace {
    fn flux_space_modes(
        &self,
        u: f64,
        u_x: f64,
        u_xx: f64,
        weights_space: &Vec<f64>,
        quad_points_space: &Vec<f64>,
        c: f64,
    ) -> (f64, f64, f64);
}

pub trait AnaFluxADERTime {
    fn flux_space_time_modes(
        &self,
        u: f64,
        u_x: f64,
        u_xx: f64,
        u_t: f64,
        u_xt: f64,
        u_tt: f64,
        weights_space: &Vec<f64>,
        quad_points_space: &Vec<f64>,
        quad_points_time: &Vec<f64>,
        c: f64,
    ) -> (f64, f64, f64);
}

pub trait AnaFluxFunction {
    fn ana_flux(
        &self,
        u_left: &na::DVector<f64>,
        u_right: &na::DVector<f64>,
        c: f64,
        size: usize,
        ghost_left: usize,
        ghost_right: usize,
    ) -> (na::DVector<f64>, na::DVector<f64>);
}

pub trait FluxFunction {
    fn num_flux_fo(&self, u: &na::DVector<f64>, c: f64, dx: f64, dt: f64) -> na::DVector<f64>;

    fn num_flux_reconstructed(
        &self,
        u_left: &na::DVector<f64>,
        u_right: &na::DVector<f64>,
        c: f64,
        dx: f64,
        dt: f64,
    ) -> na::DVector<f64>;
}

pub struct UwindFlux;
pub struct LaxFriedrichFlux;
pub struct RusanovFlux;

pub struct LinFlux;
pub struct BurgerFlux;

pub struct ADERLinFluxSpace;
pub struct ADERBurgerFluxSpace;

pub struct ADERLinFluxSpaceTime;
pub struct ADERBurgerFluxSpaceTime;

impl FluxFunction for UwindFlux {
    fn num_flux_fo(
        &self,
        u: &nalgebra::DVector<f64>,
        _c: f64,
        _dx: f64,
        _dt: f64,
    ) -> nalgebra::DVector<f64> {
        let length = u.len();
        if _c > 0.0 {
            let f: Vec<f64> = u.iter().take(length - 1).map(|&x| _c * x).collect();
            na::DVector::from_vec(f)
        } else {
            let f: Vec<f64> = u.iter().skip(1).take(length - 1).map(|&x| _c * x).collect();
            na::DVector::from_vec(f)
        }
    }

    fn num_flux_reconstructed(
        &self,
        _u_left: &nalgebra::DVector<f64>,
        _u_right: &nalgebra::DVector<f64>,
        _c: f64,
        _dx: f64,
        _dt: f64,
    ) -> nalgebra::DVector<f64> {
        panic!("not feasible for upwind!")
    }
}

impl FluxFunction for LaxFriedrichFlux {
    fn num_flux_fo(
        &self,
        u: &nalgebra::DVector<f64>,
        c: f64,
        _dx: f64,
        _dt: f64,
    ) -> nalgebra::DVector<f64> {
        let f: Vec<f64> = u
            .iter()
            .zip(u.iter().skip(1))
            .map(|(&u_l, &u_r)| 0.5 * c * (u_l + u_r) - 0.5 * c.abs() * (u_r - u_l))
            .collect();
        na::DVector::from_vec(f)
    }

    fn num_flux_reconstructed(
        &self,
        u_left: &nalgebra::DVector<f64>,
        u_right: &nalgebra::DVector<f64>,
        c: f64,
        _dx: f64,
        _dt: f64,
    ) -> nalgebra::DVector<f64> {
        let flux: Vec<f64> = u_right
            .iter()
            .zip(u_left.iter().skip(1))
            .map(|(&u_l, &u_r)| 0.5 * c * (u_r + u_l) - 0.5 * c.abs() * (u_r - u_l))
            .collect();
        na::DVector::from_vec(flux)
    }
}

impl FluxFunction for RusanovFlux {
    fn num_flux_fo(
        &self,
        u: &nalgebra::DVector<f64>,
        _c: f64,
        _dx: f64,
        _dt: f64,
    ) -> nalgebra::DVector<f64> {
        // println!("hello 1\n");
        let f: Vec<f64> = u
            .iter()
            .zip(u.iter().skip(1))
            .map(|(&v, &w)| 0.25 * (v * v + w * w) - 0.5 * (v.abs()).max(w.abs()) * (w - v))
            .collect();
        na::DVector::from_vec(f)
    }

    fn num_flux_reconstructed(
        &self,
        u_left: &nalgebra::DVector<f64>,
        u_right: &nalgebra::DVector<f64>,
        _c: f64,
        _dx: f64,
        _dt: f64,
    ) -> nalgebra::DVector<f64> {
        let f: Vec<f64> = u_right
            .iter()
            .zip(u_left.iter().skip(1))
            .map(|(&v, &w)| 0.25 * (v * v + w * w) - 0.5 * (v.abs()).max(w.abs()) * (w - v))
            .collect();
        na::DVector::from_vec(f)
    }
}

impl AnaFluxFunction for LinFlux {
    fn ana_flux(
        &self,
        u_left: &na::DVector<f64>,
        u_right: &na::DVector<f64>,
        c: f64,
        size: usize,
        ghost_left: usize,
        ghost_right: usize,
    ) -> (na::DVector<f64>, na::DVector<f64>) {
        let mut tmp_left = vec![0.0; size];
        let mut tmp_right = vec![0.0; size];

        for i in ghost_left..(size - ghost_right) {
            tmp_left[i] = c * u_left[i];
            tmp_right[i] = c * u_right[i];
        }

        let f_left = na::DVector::from_vec(tmp_left);
        let f_right = na::DVector::from_vec(tmp_right);

        (f_left, f_right)
    }
}

impl AnaFluxFunction for BurgerFlux {
    fn ana_flux(
        &self,
        u_left: &na::DVector<f64>,
        u_right: &na::DVector<f64>,
        _c: f64,
        size: usize,
        ghost_left: usize,
        ghost_right: usize,
    ) -> (na::DVector<f64>, na::DVector<f64>) {
        let mut tmp_left = vec![0.0; size];
        let mut tmp_right = vec![0.0; size];

        for i in ghost_left..(size - ghost_right) {
            tmp_left[i] = 0.5 * u_left[i] * u_left[i];
            tmp_right[i] = 0.5 * u_right[i] * u_right[i];
        }

        let f_left = na::DVector::from_vec(tmp_left);
        let f_right = na::DVector::from_vec(tmp_right);

        (f_left, f_right)
    }
}

impl AnaFluxADERSpace for ADERLinFluxSpace {
    fn flux_space_modes(
        &self,
        u: f64,
        u_x: f64,
        u_xx: f64,
        weights_space: &Vec<f64>,
        quad_points_space: &Vec<f64>,
        c: f64,
    ) -> (f64, f64, f64) {
        let mut f_0 = 0.0;
        let mut f_1 = 0.0;
        let mut f_2 = 0.0;

        for j in 0..3 {
            let space_val = quad_points_space[j];
            let space_weight = weights_space[j];

            let phi_1 = space_val;
            let phi_2 = space_val * space_val - 1.0 / 12.0;

            let val = u + u_x * phi_1 + u_xx * phi_2;

            f_0 += c * space_weight * val;
            f_1 += c * space_weight * val * phi_1;
            f_2 += c * space_weight * val * phi_2;
        }
        (f_0, f_1, f_2)
    }
}

impl AnaFluxADERSpace for ADERBurgerFluxSpace {
    fn flux_space_modes(
        &self,
        u: f64,
        u_x: f64,
        u_xx: f64,
        weights_space: &Vec<f64>,
        quad_points_space: &Vec<f64>,
        _c: f64,
    ) -> (f64, f64, f64) {
        let mut f_0 = 0.0;
        let mut f_1 = 0.0;
        let mut f_2 = 0.0;

        for j in 0..3 {
            let space_val = quad_points_space[j];
            let space_weight = weights_space[j];

            let phi_1 = space_val;
            let phi_2 = space_val * space_val - 1.0 / 12.0;

            let val = u + u_x * phi_1 + u_xx * phi_2;

            f_0 += 0.5 * space_weight * val * val;
            f_1 += 0.5 * space_weight * val * val * phi_1;
            f_2 += 0.5 * space_weight * val * val * phi_2;
        }
        (f_0, f_1, f_2)
    }
}

impl AnaFluxADERTime for ADERLinFluxSpaceTime {
    fn flux_space_time_modes(
        &self,
        u: f64,
        u_x: f64,
        u_xx: f64,
        u_t: f64,
        u_xt: f64,
        u_tt: f64,
        weights_space: &Vec<f64>,
        quad_points_space: &Vec<f64>,
        quad_points_time: &Vec<f64>,
        c: f64,
    ) -> (f64, f64, f64) {
        let mut f_3 = 0.0;
        let mut f_4 = 0.0;
        let mut f_5 = 0.0;

        for i in 0..2 {
            let time_val = quad_points_time[i] - 0.5;
            for j in 0..3 {
                let space_val = quad_points_space[j];
                let space_weight = weights_space[j];

                let phi_1 = space_val;
                let phi_2 = space_val * space_val - 1.0 / 12.0;
                let phi_3 = time_val;
                let phi_4 = time_val * space_val;
                let phi_5 = time_val * time_val - 1.0 / 12.0;

                let val =
                    u + u_x * phi_1 + u_xx * phi_2 + u_t * phi_3 + u_xt * phi_4 + u_tt * phi_5;

                f_3 += c * space_weight * val * phi_3;
                f_4 += c * space_weight * val * phi_4;
                f_5 += c * space_weight * val * phi_5;
            }

            f_3 *= 0.5;
            f_4 *= 0.5;
            f_5 *= 0.5;
        }

        (f_3, f_4, f_5)
    }
}

impl AnaFluxADERTime for ADERBurgerFluxSpaceTime {
    fn flux_space_time_modes(
        &self,
        u: f64,
        u_x: f64,
        u_xx: f64,
        u_t: f64,
        u_xt: f64,
        u_tt: f64,
        weights_space: &Vec<f64>,
        quad_points_space: &Vec<f64>,
        quad_points_time: &Vec<f64>,
        _c: f64,
    ) -> (f64, f64, f64) {
        let mut f_3 = 0.0;
        let mut f_4 = 0.0;
        let mut f_5 = 0.0;

        for i in 0..2 {
            let time_val = quad_points_time[i] - 0.5;
            for j in 0..3 {
                let space_val = quad_points_space[j];
                let space_weight = weights_space[j];

                let phi_1 = space_val;
                let phi_2 = space_val * space_val - 1.0 / 12.0;
                let phi_3 = time_val;
                let phi_4 = time_val * space_val;
                let phi_5 = time_val * time_val - 1.0 / 12.0;

                let val =
                    u + u_x * phi_1 + u_xx * phi_2 + u_t * phi_3 + u_xt * phi_4 + u_tt * phi_5;

                f_3 += 0.5 * space_weight * val * val * phi_3;
                f_4 += 0.5 * space_weight * val * val * phi_4;
                f_5 += 0.5 * space_weight * val * val * phi_5;
            }

            f_3 *= 0.5;
            f_4 *= 0.5;
            f_5 *= 0.5;
        }

        (f_3, f_4, f_5)
    }
}
