pub mod convergence;
pub mod examples;
pub mod fv_core;
pub mod time_integrator;

use examples::{ader_third_lin, conv_burger, pc_lin};
use std::f64::consts::PI;

fn analytical(x: f64) -> f64 {
    (2.0 * PI * x).sin()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ader_third_lin()?;
    // pc_lin()?;
    // conv_burger()?;
    Ok(())
}
