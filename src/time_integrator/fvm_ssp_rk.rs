extern crate nalgebra as na;

use crate::fv_core::{
    condition::{initialize_mesh, BCEnforcer, InitialCondition},
    flux::FluxFunction,
    mesh::Mesh,
    reconstruction::Reconstruction,
};

use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::error::Error;

//TRAITS
pub trait ConservationLaw {
    fn max_speed(&self, u: &na::DVector<f64>) -> f64;
}

pub trait TimeIntegrator {
    fn update(
        &self,
        u: &na::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        recon: Option<&dyn Reconstruction>,
    ) -> na::DVector<f64>;
}

//STRUCTS
pub struct LinearAdvection {
    pub c: f64,
}
pub struct Burger;

pub struct FoIntegrator;
pub struct SspRk2Integrator;
pub struct SspRk3IntegratorPPM;
pub struct SspRk3IntegratorWENO;

//IMPLEMENTATIONS
impl ConservationLaw for LinearAdvection {
    fn max_speed(&self, _u: &nalgebra::DVector<f64>) -> f64 {
        self.c.abs()
    }
}

impl ConservationLaw for Burger {
    fn max_speed(&self, u: &nalgebra::DVector<f64>) -> f64 {
        u.iter().fold(0.0, |max, &v| max.max(v.abs()))
    }
}

impl TimeIntegrator for FoIntegrator {
    fn update(
        &self,
        u: &nalgebra::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        recon: Option<&dyn Reconstruction>,
    ) -> nalgebra::DVector<f64> {
        let mut u_new = u.clone();

        let f = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(u);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(u, c, dx, dt)
        };

        for i in ghost_left..(size - ghost_right) {
            u_new[i] = u[i] - dt / dx * (f[i] - f[i - 1]);
        }
        bc.enforce(&mut u_new);

        u_new
    }
}

impl TimeIntegrator for SspRk2Integrator {
    fn update(
        &self,
        u: &nalgebra::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        recon: Option<&dyn Reconstruction>,
    ) -> nalgebra::DVector<f64> {
        let mut u_new = u.clone();

        let f1 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(u);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(u, c, dx, dt)
        };

        let mut u1 = u.clone();
        for i in ghost_left..(size - ghost_right) {
            u1[i] = u[i] - dt / dx * (f1[i] - f1[i - 1]);
        }
        bc.enforce(&mut u1);

        let f2 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(&u1);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(&u1, c, dx, dt)
        };

        for i in ghost_left..(size - ghost_right) {
            u_new[i] = 0.5 * u[i] + 0.5 * u1[i] - 0.5 * dt / dx * (f2[i] - f2[i - 1]);
        }
        bc.enforce(&mut u_new);

        u_new
    }
}

impl TimeIntegrator for SspRk3IntegratorPPM {
    fn update(
        &self,
        u: &nalgebra::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        recon: Option<&dyn Reconstruction>,
    ) -> nalgebra::DVector<f64> {
        let mut u_new = u.clone();

        //stage 1
        let f1 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(u);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(u, c, dx, dt)
        };

        let mut u1 = u.clone();
        for i in ghost_left..(size - ghost_right) {
            u1[i] = u[i] - dt / dx * (f1[i] - f1[i - 1]);
        }
        bc.enforce(&mut u1);

        //stage 2
        let f2 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(&u1);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(&u1, c, dx, dt)
        };

        let mut u2 = u.clone();
        for i in ghost_left..(size - ghost_right) {
            u2[i] = (3.0 / 4.0) * u[i] + (1.0 / 4.0) * u1[i]
                - (1.0 / 4.0) * dt / dx * (f2[i] - f2[i - 1]);
        }
        bc.enforce(&mut u2);

        //stage 3
        let f3 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(&u2);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(&u2, c, dx, dt)
        };

        for i in ghost_left..(size - ghost_right) {
            u_new[i] = (1.0 / 3.0) * u[i] + (2.0 / 3.0) * u2[i]
                - (2.0 / 3.0) * dt / dx * (f3[i] - f3[i - 1]);
        }
        bc.enforce(&mut u_new);

        u_new
    }
}

impl TimeIntegrator for SspRk3IntegratorWENO {
    fn update(
        &self,
        u: &nalgebra::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        recon: Option<&dyn Reconstruction>,
    ) -> nalgebra::DVector<f64> {
        let mut u_new = u.clone();

        //stage 1
        let f1 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(u);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(u, c, dx, dt)
        };

        let mut u1 = u.clone();
        for i in ghost_left..(size - ghost_right) {
            u1[i] = u[i] - dt / dx * (f1[i - 1] - f1[i - 2]);
        }
        bc.enforce(&mut u1);

        //stage 2
        let f2 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(&u1);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(&u1, c, dx, dt)
        };

        let mut u2 = u.clone();
        for i in ghost_left..(size - ghost_right) {
            u2[i] = (3.0 / 4.0) * u[i] + (1.0 / 4.0) * u1[i]
                - (1.0 / 4.0) * dt / dx * (f2[i - 1] - f2[i - 2]);
        }
        bc.enforce(&mut u2);

        //stage 3
        let f3 = if let Some(recon) = recon {
            let (u_left, u_right) = recon.reconstruction(&u2);
            flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt)
        } else {
            flux_func.num_flux_fo(&u2, c, dx, dt)
        };

        for i in ghost_left..(size - ghost_right) {
            u_new[i] = (1.0 / 3.0) * u[i] + (2.0 / 3.0) * u2[i]
                - (2.0 / 3.0) * dt / dx * (f3[i - 1] - f3[i - 2]);
        }
        bc.enforce(&mut u_new);

        u_new
    }
}

//HELPERS
#[derive(Serialize)]
struct RowDataAna {
    x: f64,
    u_initial: f64,
    u_final: f64,
    u_analytical: f64,
}

#[derive(Serialize)]
struct RowData {
    x: f64,
    u_initial: f64,
    u_final: f64,
}

fn periodic_wrap(x: f64, x_min: f64, x_max: f64) -> f64 {
    let len = x_max - x_min;
    let mut y = (x - x_min) % len;
    if y < 0.0 {
        y += len;
    }
    x_min + y
}

fn calc_time_step(dx: f64, cfl: f64, s_max: f64) -> f64 {
    cfl * dx / s_max
}

pub fn solver_ssp_rk<M, I, F, R, BC, CL, TI>(
    mesh: &M,
    init: &I,
    flux_func: &F,
    recon: Option<R>,
    bc: &BC,
    law: &CL,
    integrator: TI,
    c: f64,
    cfl: f64,
    t_begin: f64,
    t_end: f64,
    csv_path: &str,
    x_begin: f64,
    x_end: f64,
    analytical: Option<&dyn Fn(f64) -> f64>,
) -> Result<(), Box<dyn Error>>
where
    M: Mesh,
    I: InitialCondition,
    F: FluxFunction,
    R: Reconstruction,
    BC: BCEnforcer,
    CL: ConservationLaw,
    TI: TimeIntegrator,
{
    let dx = mesh.get_dx();
    let mut dt;
    let mut u = initialize_mesh(mesh, init, bc);
    let u_init = u.clone();

    let total_time = t_end - t_begin;
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}% (eta: {eta}) {msg}",
            )
            .unwrap()
            .progress_chars("█░"),
    );

    let ghost_left = mesh.ghost_left();
    let ghost_right = mesh.ghost_right();
    let size = mesh.get_length();

    let mut t = t_begin;
    while t < t_end {
        dt = calc_time_step(dx, cfl, law.max_speed(&u));

        u = integrator.update(
            &u,
            dx,
            dt,
            c,
            ghost_left,
            ghost_right,
            size,
            flux_func,
            bc,
            recon.as_ref().map(|r| r as &dyn Reconstruction),
        );

        t += dt;
        let fraction = (t - t_begin) / total_time;
        pb.set_position((fraction * 100.0) as u64);
    }
    dt = t_end - t;
    u = integrator.update(
        &u,
        dx,
        dt,
        c,
        ghost_left,
        ghost_right,
        size,
        flux_func,
        bc,
        recon.as_ref().map(|r| r as &dyn Reconstruction),
    );

    pb.finish_with_message("\nSimulation complete");

    let mut wtr = Writer::from_path(csv_path)?;

    for i in ghost_left..(size - ghost_right) {
        let x = mesh.get_cell(i);

        if let Some(analytical_fn) = analytical {
            let x_ana = periodic_wrap(x - c * t_end, x_begin, x_end);
            let ana_val = analytical_fn(x_ana);
            let row = RowDataAna {
                x,
                u_initial: u_init[i],
                u_final: u[i],
                u_analytical: ana_val,
            };
            wtr.serialize(row)?;
        } else {
            let row = RowData {
                x,
                u_initial: u_init[i],
                u_final: u[i],
            };
            wtr.serialize(row)?;
        }
    }
    wtr.flush()?;

    Ok(())
}
