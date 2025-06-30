pub mod convergence;
pub mod examples;
pub mod fv_core;
pub mod time_integrator;

use examples::{
    ader_third_lin, conv_burger, conv_burger_ader3, conv_burger_mp_pc, conv_burger_pc,
    conv_lin_adv_ader3, conv_lin_mp_pc, conv_lin_pc, pc_lin,
};
use std::f64::consts::PI;

fn analytical(x: f64) -> f64 {
    (2.0 * PI * x).sin()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ader_third_lin()?;
    // pc_lin()?;
    // conv_burger()?;
    // conv_lin_pc()?;
    // conv_lin_mp_pc()?;
    // conv_burger_pc()?;
    // conv_burger_mp_pc()?;
    conv_lin_adv_ader3()?;
    // conv_burger_ader3()?;
    Ok(())
}
