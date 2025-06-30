extern crate nalgebra as na;

use crate::fv_core::{
    condition::{initialize_mesh, BCEnforcer, InitialCondition},
    flux::{AnaFluxADERSpace, AnaFluxADERTime, AnaFluxFunction, FluxFunction},
    mesh::Mesh,
    reconstruction::{Reconstruction, WENOReconstruction, WENOReconstructionMode},
};

use crate::time_integrator::fvm_ssp_rk::ConservationLaw;

use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;
use std::error::Error;

//TRAITS
pub trait TimeIntegratorADER {
    fn update(
        &self,
        u: &na::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_ana: &dyn AnaFluxFunction,
        ader_flux_space: &dyn AnaFluxADERSpace,
        ader_flux_time: &dyn AnaFluxADERTime,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        recon: Option<&dyn Reconstruction>,
    ) -> na::DVector<f64>;
}

pub trait TimeIntegratorMaxPrincipal {
    fn update(
        &self,
        min: f64,
        max: f64,
        u: &na::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_ana: &dyn AnaFluxFunction,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        recon: Option<&dyn Reconstruction>,
    ) -> na::DVector<f64>;
}

//STRUCTS
pub struct PredictorCorrector;
pub struct MPPredictorCorrector;
pub struct ADER1DThirdOrder;

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

fn calc_time_step_mp(dx: f64, s_max: f64) -> f64 {
    let cfl = 1.0 / 6.0;

    cfl * dx / s_max
}

fn limiter(
    max_ana: f64,
    max_pol: &Vec<f64>,
    min_ana: f64,
    min_pol: &Vec<f64>,
    u: &na::DVector<f64>,
    size: usize,
    ghost_l: usize,
    ghost_r: usize,
) -> Vec<f64> {
    let mut tmp = vec![0.0; size];

    for i in ghost_l..(size - ghost_r) {
        let tmp1 = (max_ana - u[i]) / (max_pol[i] - u[i]);
        let tmp2 = (min_ana - u[i]) / (min_pol[i] - u[i]);

        let tmp3 = (tmp1.abs()).min(tmp2.abs());

        tmp[i] = tmp3.min(1.0);
    }

    tmp
}
//IMPLEMENTAITIONS
impl TimeIntegratorADER for PredictorCorrector {
    fn update(
        &self,
        u: &nalgebra::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_ana: &dyn AnaFluxFunction,
        _ader_flux_space: &dyn AnaFluxADERSpace,
        _ader_flux_time: &dyn AnaFluxADERTime,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        _recon: Option<&dyn Reconstruction>,
    ) -> nalgebra::DVector<f64> {
        //reconstruct modes
        let recon_pc = WENOReconstruction {
            gamma_l: 1.0,
            gamma_c: 50.0,
            gamma_r: 1.0,
            p: 4,
        };
        let (u_x, _u_xx) = recon_pc.compute_modes(&u);

        //predictor step
        let mut u_t: na::DVector<f64> = na::DVector::zeros(size);
        let mut u_left_p: na::DVector<f64> = na::DVector::zeros(size);
        let mut u_right_p: na::DVector<f64> = na::DVector::zeros(size);

        for i in ghost_left..(size - ghost_right) {
            u_left_p[i] = u[i] - 0.5 * u_x[i];
            u_right_p[i] = u[i] + 0.5 * u_x[i];
        }

        let (f_left, f_right) =
            flux_ana.ana_flux(&u_left_p, &u_right_p, c, size, ghost_left, ghost_right);

        for i in ghost_left..(size - ghost_right) {
            u_t[i] = -1.0 * dt / dx * (f_right[i] - f_left[i]);
        }

        let u_left_phys: Vec<f64> = (2..(size - ghost_right))
            .map(|i| u[i] - 0.5 * u_x[i] + 0.5 * u_t[i])
            .collect();
        let u_right_phys: Vec<f64> = (2..(size - ghost_right))
            .map(|i| u[i] + 0.5 * u_x[i] + 0.5 * u_t[i])
            .collect();
        let mut tmp_left = Vec::with_capacity(size - 2);
        let mut tmp_right = Vec::with_capacity(size - 2);

        tmp_left.push(0.0);
        tmp_left.extend(u_left_phys);
        tmp_left.push(0.0);

        tmp_left[0] = tmp_left[size - 4];
        tmp_left[size - 3] = tmp_left[1];

        tmp_right.push(0.0);
        tmp_right.extend(u_right_phys);
        tmp_right.push(0.0);

        tmp_right[0] = tmp_right[size - 4];
        tmp_right[size - 3] = tmp_right[1];

        let u_left = na::DVector::from_vec(tmp_left);
        let u_right = na::DVector::from_vec(tmp_right);

        //corrector step
        let f = flux_func.num_flux_reconstructed(&u_left, &u_right, c, dx, dt);

        let mut u_new = na::DVector::zeros(size);
        for i in ghost_left..(size - ghost_right) {
            u_new[i] = u[i] - dt / dx * (f[i - 1] - f[i - 2]);
        }
        bc.enforce(&mut u_new);

        u_new
    }
}

impl TimeIntegratorADER for ADER1DThirdOrder {
    fn update(
        &self,
        u: &na::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        _flux_ana: &dyn AnaFluxFunction,
        ader_flux_space: &dyn AnaFluxADERSpace,
        ader_flux_time: &dyn AnaFluxADERTime,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        _recon: Option<&dyn Reconstruction>,
    ) -> na::DVector<f64> {
        let recon = WENOReconstruction {
            gamma_l: 1.0,
            gamma_c: 50.0,
            gamma_r: 1.0,
            p: 4,
        };
        //my ader
        let (u_1, u_2) = recon.compute_modes(&u);

        let weights = vec![1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0];
        let quad_points_space = vec![-0.5, 0.0, 0.5];

        let q_tmp_1 = 0.5 - 0.5 * ((1.0 / 3.0) as f64).sqrt();
        let q_tmp_2 = 0.5 + 0.5 * ((1.0 / 3.0) as f64).sqrt();

        let quad_points_time = vec![q_tmp_1, q_tmp_2];

        let mut u_3 = na::DVector::<f64>::zeros(size);
        let mut u_4 = na::DVector::<f64>::zeros(size);
        let mut u_5 = na::DVector::<f64>::zeros(size);

        for i in ghost_left..(size - ghost_right) {
            let (_f_0, f_1, f_2) = ader_flux_space.flux_space_modes(
                u[i],
                u_1[i],
                u_2[i],
                &weights,
                &quad_points_space,
                c,
            );

            for _ in 0..3 {
                let (_f_3, f_4, _f_5) = ader_flux_time.flux_space_time_modes(
                    u[i],
                    u_1[i],
                    u_2[i],
                    u_3[i],
                    u_4[i],
                    u_5[i],
                    &weights,
                    &quad_points_space,
                    &quad_points_time,
                    c,
                );

                u_3[i] = -1.0 * dt / dx * f_1;
                u_4[i] = -2.0 * dt / dx * f_2;
                u_5[i] = -0.5 * dt / dx * f_4;
            }
        }

        //balsara ader
        /*let q_1 = 0.5 - 0.5 * ((1.0 / 3.0) as f64).sqrt();
        let q_2 = 0.5 + 0.5 * ((1.0 / 3.0) as f64).sqrt();

        let mut u_3 = na::DVector::<f64>::zeros(size);
        let mut u_4 = na::DVector::<f64>::zeros(size);
        let mut u_5 = na::DVector::<f64>::zeros(size);

        for i in ghost_left..(size - ghost_right) {
            let u_tilde_0 = u[i];
            let u_tilde_1 = u[i] + 0.5 * u_1[i] + u_2[i] * 1.0 / 6.0;
            let u_tilde_2 = u[i] - 0.5 * u_1[i] + u_2[i] * 1.0 / 6.0;

            let f_tmp_0 = c * u_tilde_0;
            let f_tmp_1 = c * u_tilde_1;
            let f_tmp_2 = c * u_tilde_2;

            let _f_0 = (4.0 * f_tmp_0 + f_tmp_2 + f_tmp_2) / 6.0;
            let f_1 = f_tmp_1 - f_tmp_2;
            let f_2 = 2.0 * (f_tmp_1 - 2.0 * f_tmp_0 + f_tmp_2);

            for _ in 0..3 {
                let u_tilde_3 = u[i]
                    + 0.5 * u_1[i]
                    + 1.0 / 6.0 * u_2[i]
                    + 0.5 * u_3[i]
                    + 0.25 * u_4[i]
                    + 0.25 * u_5[i];
                let u_tilde_4 = u[i] - 0.5 * u_1[i] + 1.0 / 6.0 * u_2[i] + 0.5 * u_3[i]
                    - 0.25 * u_4[i]
                    + 0.25 * u_5[i];
                let u_tilde_5 = u[i] + u_3[i] + u_5[i];

                let f_tmp_3 = c * u_tilde_3;
                let f_tmp_4 = c * u_tilde_4;
                let f_tmp_5 = c * u_tilde_5;

                let _f_3 = f_tmp_0 - 2.0 * f_tmp_1 - 2.0 * f_tmp_2 + 2.0 * f_tmp_3 + 2.0 * f_tmp_4
                    - f_tmp_5;
                let f_4 = -2.0 * (f_tmp_1 - f_tmp_2 - f_tmp_3 + f_tmp_4);
                let _f_5 = -2.0 * (f_tmp_0 - f_tmp_1 - f_tmp_2 + f_tmp_3 + f_tmp_4 - f_tmp_5);

                u_3[i] = -1.0 * dt / dx * f_1;
                u_4[i] = -2.0 * dt / dx * f_2;
                u_5[i] = -0.5 * dt / dx * f_4;
            }
        }*/

        let mut u_minus_1_tmp = na::DVector::<f64>::zeros(size);
        let mut u_plus_1_tmp = na::DVector::<f64>::zeros(size);

        let mut u_minus_2_tmp = na::DVector::<f64>::zeros(size);
        let mut u_plus_2_tmp = na::DVector::<f64>::zeros(size);

        for i in ghost_left..(size - ghost_right) {
            let q_1 = q_tmp_1 - 0.5;
            let q_2 = q_tmp_2 - 0.5;

            u_plus_1_tmp[i] = u[i] - 0.5 * u_1[i] + 1.0 / 6.0 * u_2[i] + q_1 * u_3[i]
                - 0.5 * q_1 * u_4[i]
                + q_1 * q_1 * u_5[i];
            u_minus_1_tmp[i] = u[i]
                + 0.5 * u_1[i]
                + 1.0 / 6.0 * u_2[i]
                + q_1 * u_3[i]
                + 0.5 * q_1 * u_4[i]
                + q_1 * q_1 * u_5[i];

            u_plus_2_tmp[i] = u[i] - 0.5 * u_1[i] + 1.0 / 6.0 * u_2[i] + q_2 * u_3[i]
                - 0.5 * q_2 * u_4[i]
                + q_2 * q_2 * u_5[i];
            u_minus_2_tmp[i] = u[i]
                + 0.5 * u_1[i]
                + 1.0 / 6.0 * u_2[i]
                + q_2 * u_3[i]
                + 0.5 * q_2 * u_4[i]
                + q_2 * q_2 * u_5[i];
        }

        bc.enforce(&mut u_minus_1_tmp);
        bc.enforce(&mut u_plus_1_tmp);
        bc.enforce(&mut u_minus_2_tmp);
        bc.enforce(&mut u_plus_2_tmp);

        let u_minus_1 = u_minus_1_tmp.remove_row(size - 1).remove_row(0);
        let u_plus_1 = u_plus_1_tmp.remove_row(size - 1).remove_row(0);

        let u_minus_2 = u_minus_2_tmp.remove_row(size - 1).remove_row(0);
        let u_plus_2 = u_plus_2_tmp.remove_row(size - 1).remove_row(0);

        let f_1 = flux_func.num_flux_reconstructed(&u_plus_1, &u_minus_1, c, dx, dt);
        let f_2 = flux_func.num_flux_reconstructed(&u_plus_2, &u_minus_2, c, dx, dt);

        let mut u_new = na::DVector::<f64>::zeros(size);
        for i in ghost_left..(size - ghost_right) {
            u_new[i] = u[i]
                - 0.5 * dt / dx * (f_1[i - 1] - f_1[i - 2])
                - 0.5 * dt / dx * (f_2[i - 1] - f_2[i - 2]);
        }

        bc.enforce(&mut u_new);

        u_new
    }
}

impl TimeIntegratorMaxPrincipal for MPPredictorCorrector {
    fn update(
        &self,
        min: f64,
        max: f64,
        u: &na::DVector<f64>,
        dx: f64,
        dt: f64,
        c: f64,
        ghost_left: usize,
        ghost_right: usize,
        size: usize,
        flux_ana: &dyn AnaFluxFunction,
        flux_func: &dyn FluxFunction,
        bc: &dyn BCEnforcer,
        _recon: Option<&dyn Reconstruction>,
    ) -> na::DVector<f64> {
        let mut limit_one;
        let mut limit_two;

        let recon_ader = WENOReconstruction {
            gamma_l: 1.0,
            gamma_c: 50.0,
            gamma_r: 1.0,
            p: 4,
        };

        let mut u_plus_zero = na::DVector::zeros(size);
        let mut u_minus_zero = na::DVector::zeros(size);

        let mut u_plus_half = na::DVector::zeros(size);
        let mut u_minus_half = na::DVector::zeros(size);

        let mut min_pol1 = vec![0.0; size];
        let mut max_pol1 = vec![0.0; size];
        let mut min_pol2 = vec![0.0; size];
        let mut max_pol2 = vec![0.0; size];

        let (mut u_x, _u_xx) = recon_ader.compute_modes(&u);
        let mut u_t = na::DVector::zeros(size);

        loop {
            for i in ghost_left..(size - ghost_right) {
                u_plus_zero[i] = u[i] - 0.5 * u_x[i];
                u_minus_zero[i] = u[i] + 0.5 * u_x[i];

                min_pol1[i] = u_plus_zero[i].min(u_minus_zero[i]);
                max_pol1[i] = u_plus_zero[i].max(u_minus_zero[i]);
            }

            limit_one = limiter(
                max,
                &max_pol1,
                min,
                &min_pol1,
                &u,
                size,
                ghost_left,
                ghost_right,
            );

            limit_one[0] = limit_one[size - 4];
            limit_one[1] = limit_one[size - 3];
            limit_one[size - 2] = limit_one[2];
            limit_one[size - 1] = limit_one[3];

            for i in ghost_left..(size - ghost_right) {
                u_x[i] *= limit_one[i];
            }

            let (flux_plus, flux_minus) = flux_ana.ana_flux(
                &u_plus_zero,
                &u_minus_zero,
                c,
                size,
                ghost_left,
                ghost_right,
            );

            for i in ghost_left..(size - ghost_right) {
                u_t[i] = -1.0 / dx * (flux_minus[i] - flux_plus[i]);

                u_plus_half[i] = u[i] - 0.5 * u_x[i] + 0.5 * dt * u_t[i];
                u_minus_half[i] = u[i] + 0.5 * u_x[i] + 0.5 * dt * u_t[i];

                min_pol2[i] = u_plus_half[i].min(u_minus_half[i]);
                max_pol2[i] = u_plus_half[i].max(u_minus_half[i]);
            }

            limit_two = limiter(
                max,
                &max_pol2,
                min,
                &min_pol2,
                &u,
                size,
                ghost_left,
                ghost_right,
            );

            for i in ghost_left..(size - ghost_right) {
                u_x[i] *= limit_two[i];
            }

            limit_two[0] = limit_two[size - 4];
            limit_two[1] = limit_two[size - 3];
            limit_two[size - 2] = limit_two[2];
            limit_two[size - 1] = limit_two[3];

            if limit_one.iter().all(|&xi| xi == 1.0) && limit_two.iter().all(|&xi| xi == 1.0) {
                break;
            }
        }

        let mut u_plus = u_plus_half.remove_row(size - 1).remove_row(0);
        let mut u_minus = u_minus_half.remove_row(size - 1).remove_row(0);
        let length = u_plus.len();

        u_plus[0] = u_plus[length - 2];
        u_plus[length - 1] = u_plus[1];

        u_minus[0] = u_minus[length - 2];
        u_minus[length - 1] = u_minus[1];

        let f = flux_func.num_flux_reconstructed(&u_plus, &u_minus, c, dx, dt);
        let mut u_new = na::DVector::zeros(size);

        for i in ghost_left..(size - ghost_right) {
            u_new[i] = u[i] - dt / dx * (f[i - 1] - f[i - 2]);
        }

        bc.enforce(&mut u_new);

        u_new
    }
}

pub fn solver_ader<M, I, FN, FA, AFS, AFT, R, BC, CL, TI>(
    mesh: &M,
    init: &I,
    flux_func: &FN,
    flux_ana: &FA,
    ader_flux_space: &AFS,
    ader_flux_time: &AFT,
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
    FN: FluxFunction,
    FA: AnaFluxFunction,
    AFS: AnaFluxADERSpace,
    AFT: AnaFluxADERTime,
    R: Reconstruction,
    BC: BCEnforcer,
    CL: ConservationLaw,
    TI: TimeIntegratorADER,
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
            flux_ana,
            ader_flux_space,
            ader_flux_time,
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
        flux_ana,
        ader_flux_space,
        ader_flux_time,
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

pub fn solver_ader_max_principal<M, I, FN, FA, R, BC, CL, TI>(
    mesh: &M,
    init: &I,
    flux_func: &FN,
    flux_ana: &FA,
    recon: Option<R>,
    bc: &BC,
    law: &CL,
    integrator: TI,
    c: f64,
    _cfl: f64,
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
    FN: FluxFunction,
    FA: AnaFluxFunction,
    R: Reconstruction,
    BC: BCEnforcer,
    CL: ConservationLaw,
    TI: TimeIntegratorMaxPrincipal,
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

    let min = u.min();
    let max = u.max();

    let mut t = t_begin;
    while t < t_end {
        dt = calc_time_step_mp(dx, law.max_speed(&u));

        u = integrator.update(
            min,
            max,
            &u,
            dx,
            dt,
            c,
            ghost_left,
            ghost_right,
            size,
            flux_ana,
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
        min,
        max,
        &u,
        dx,
        dt,
        c,
        ghost_left,
        ghost_right,
        size,
        flux_ana,
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
