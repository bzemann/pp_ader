use std::f64::consts::PI;

use crate::fv_core::condition::{
    PPMInit,
    StandardInit,
    WENOInit, // your definitions
};

/// A sine wave initial condition for StandardInit.
pub fn sin_standard() -> StandardInit<fn(f64) -> f64> {
    // Here, the closure doesn't capture anything, so we can coerce it to a function pointer.
    StandardInit {
        f: (|x| (2.0 * PI * x).sin()) as fn(f64) -> f64,
    }
}

/// A Gaussian initial condition for StandardInit, parameterized by mu and sigma.
/// Here we use a boxed closure, because we need to capture `mu` and `sigma`.
pub fn gauss_standard(mu: f64, sigma: f64) -> StandardInit<Box<dyn Fn(f64) -> f64>> {
    StandardInit {
        f: Box::new(move |x| {
            let exponent = -((x - mu) / (2.0 * sigma)).powi(2);
            exponent.exp()
        }),
    }
}

/// A square-box initial condition for StandardInit, set to 0.9 in [x_left, x_right], else 0.0.
pub fn square_box_standard(x_left: f64, x_right: f64) -> StandardInit<Box<dyn Fn(f64) -> f64>> {
    StandardInit {
        f: Box::new(move |x| {
            if x >= x_left && x <= x_right {
                0.9
            } else {
                0.0
            }
        }),
    }
}

// Repeat similarly for PPMInit:
pub fn sin_ppm() -> PPMInit<fn(f64) -> f64> {
    PPMInit {
        f: (|x| (2.0 * PI * x).sin()) as fn(f64) -> f64,
    }
}

pub fn gauss_ppm(mu: f64, sigma: f64) -> PPMInit<Box<dyn Fn(f64) -> f64>> {
    PPMInit {
        f: Box::new(move |x| {
            let exponent = -((x - mu) / (2.0 * sigma)).powi(2);
            exponent.exp()
        }),
    }
}

pub fn square_box_ppm(x_left: f64, x_right: f64) -> PPMInit<Box<dyn Fn(f64) -> f64>> {
    PPMInit {
        f: Box::new(move |x| {
            if x >= x_left && x <= x_right {
                0.9
            } else {
                0.0
            }
        }),
    }
}

// And for WENOInit:
pub fn sin_weno() -> WENOInit<fn(f64) -> f64> {
    WENOInit {
        f: (|x| (2.0 * PI * x).sin()) as fn(f64) -> f64,
    }
}

pub fn sin_mp_ex_5_2() -> WENOInit<fn(f64) -> f64> {
    WENOInit {
        f: (|x| (0.25 + 0.5 * (PI * x).sin())) as fn(f64) -> f64,
    }
}

pub fn gauss_weno(mu: f64, sigma: f64) -> WENOInit<Box<dyn Fn(f64) -> f64>> {
    WENOInit {
        f: Box::new(move |x| {
            let exponent = -((x - mu) / (2.0 * sigma)).powi(2);
            exponent.exp()
        }),
    }
}

pub fn square_box_weno(x_left: f64, x_right: f64) -> WENOInit<Box<dyn Fn(f64) -> f64>> {
    WENOInit {
        f: Box::new(move |x| {
            if x >= x_left && x <= x_right {
                0.9
            } else {
                0.0
            }
        }),
    }
}
