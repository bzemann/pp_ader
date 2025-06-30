extern crate nalgebra as na;

use crate::{
    fv_core::{
        condition::{initialize_mesh, InitialCondition, PeriodicBC, PerioidcBCWENO},
        flux::{
            ADERBurgerFluxSpace, ADERBurgerFluxSpaceTime, ADERLinFluxSpace, ADERLinFluxSpaceTime,
            BurgerFlux, LaxFriedrichFlux, LinFlux, RusanovFlux,
        },
        mesh::{Mesh, Mesh1d},
        reconstruction::{McLimiter, TVDReconstruction, WENOReconstruction},
    },
    time_integrator::{
        ader::{
            ADER1DThirdOrder, MPPredictorCorrector, PredictorCorrector, TimeIntegratorADER,
            TimeIntegratorMaxPrincipal,
        },
        fvm_ssp_rk::{
            Burger, ConservationLaw, FoIntegrator, LinearAdvection, SspRk2Integrator,
            SspRk3IntegratorWENO, TimeIntegrator,
        },
    },
};

use csv::Writer;
use serde::Serialize;
use std::error::Error;
use std::f64::consts::PI;

#[derive(Serialize)]
struct RowData {
    mesh: usize,
    err_fo: f64,
    err_tvd: f64,
    err_weno: f64,
}

#[derive(Serialize)]
struct RowData1 {
    mesh: usize,
    err: f64,
}

#[derive(Serialize)]
struct _Test {
    mesh: f64,
    u: f64,
    u_exact: f64,
}

fn calc_err(
    u_aprox: &na::DVector<f64>,
    u_exact: &na::DVector<f64>,
    size: usize,
    ghost_left: usize,
    ghost_right: usize,
    dx: f64,
) -> f64 {
    let mut err = 0.0;
    for i in ghost_left..(size - ghost_right) {
        err += dx * (u_aprox[i] - u_exact[i]).abs();
    }
    err
}

fn calc_err_burger(
    u_aprox: &na::DVector<f64>,
    u_exact: &na::DVector<f64>,
    size_approx: usize,
    size_exact: usize,
    g_l_approx: usize,
    g_r_approx: usize,
    g_l_exact: usize,
    g_r_exact: usize,
    dx: f64,
) -> f64 {
    let n_aprox = size_approx - (g_l_approx + g_r_approx);
    let n_exact = size_exact - (g_l_exact + g_r_exact);

    let refinement_ratio = n_exact / n_aprox;
    let sampled_exact: Vec<f64> = (0..n_aprox)
        .map(|i| {
            let start = 2 + i * refinement_ratio;
            let end = start + refinement_ratio;
            let sum: f64 = (start..end).map(|j| u_exact[j]).sum();
            sum / refinement_ratio as f64
        })
        .collect();

    let mut err = 0.0;
    for i in g_l_approx..(size_approx - g_r_approx) {
        err += dx * (u_aprox[i] - sampled_exact[i - g_l_approx]).abs();
    }
    err
}

fn calc_time_step(dx: f64, cfl: f64, s_max: f64) -> f64 {
    cfl * dx / s_max
}

fn periodic_wrap(x: f64, x_min: f64, x_max: f64) -> f64 {
    let length = x_max - x_min;
    let mut y = (x - x_min) % length;
    if y < 0.0 {
        y += length;
    }
    x_min + y
}

pub fn convergence_lin_adv<IS, IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_std: &IS,
    init_weno: &IW,
    cfl_fo: f64,
    cfl_tvd: f64,
    cfl_weno: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IS: InitialCondition,
    IW: InitialCondition,
{
    let ana_int = |x: f64| -> f64 { (2.0 * PI * x).cos() };

    let base_points = 20;
    let mesh_sizes: Vec<usize> = (0..8).map(|i| base_points * 2_usize.pow(i)).collect();
    let num_meshes = mesh_sizes.len();

    let law = LinearAdvection { c };
    let flux_func = LaxFriedrichFlux;

    let mut wtr = Writer::from_path(csv_path)?;

    for i in 0..num_meshes {
        let current_mesh_size = mesh_sizes[i];
        println!("current mesh: {}", current_mesh_size);

        //fo
        let mesh_fo = Mesh1d::new(x_begin, x_end, current_mesh_size, 1);
        let dx_fo = mesh_fo.get_dx();
        let size_fo = mesh_fo.get_length();
        let g_left_fo = mesh_fo.ghost_left();
        let g_right_fo = mesh_fo.ghost_right();

        let bc_fo = PeriodicBC;
        let integrator_fo = FoIntegrator;

        let mut u_fo = initialize_mesh(&mesh_fo, init_std, &bc_fo);
        let mut u_exact_fo = u_fo.clone();

        let s_max = law.max_speed(&u_fo);
        let mut dt_fo = calc_time_step(dx_fo, cfl_fo, s_max);
        let mut t = t_begin;

        while t < t_end {
            u_fo = integrator_fo.update(
                &u_fo, dx_fo, dt_fo, c, g_left_fo, g_right_fo, size_fo, &flux_func, &bc_fo, None,
            );

            t += dt_fo;
        }
        dt_fo = t_end - t;
        u_fo = integrator_fo.update(
            &u_fo, dx_fo, dt_fo, c, g_left_fo, g_right_fo, size_fo, &flux_func, &bc_fo, None,
        );

        for i in g_left_fo..(size_fo - g_right_fo) {
            let x = mesh_fo[i];
            let x_l = x - 0.5 * dx_fo;
            let x_r = x + 0.5 * dx_fo;
            let x_ana_l = periodic_wrap(x_l - c * t_end, x_begin, x_end);
            let x_ana_r = periodic_wrap(x_r - c * t_end, x_begin, x_end);

            u_exact_fo[i] = 1.0 / (dx_fo * 2.0 * PI) * (ana_int(x_ana_l) - ana_int(x_ana_r));
        }

        let err_fo = calc_err(&u_fo, &u_exact_fo, size_fo, g_left_fo, g_right_fo, dx_fo);

        //tvd
        let mesh_tvd = Mesh1d::new(x_begin, x_end, current_mesh_size, 1);
        let dx_tvd = mesh_tvd.get_dx();
        let size_tvd = mesh_tvd.get_length();
        let g_left_tvd = mesh_tvd.ghost_left();
        let g_right_tvd = mesh_tvd.ghost_right();

        let bc_tvd = PeriodicBC;
        let recon_tvd = TVDReconstruction { limiter: McLimiter };
        let integrator_tvd = SspRk2Integrator;

        let mut u_tvd = initialize_mesh(&mesh_tvd, init_std, &bc_tvd);
        let mut u_exact_tvd = u_tvd.clone();

        let s_max = law.max_speed(&u_tvd);
        let mut dt_tvd = calc_time_step(dx_tvd, cfl_tvd, s_max);
        t = t_begin;

        while t < t_end {
            u_tvd = integrator_tvd.update(
                &u_tvd,
                dx_tvd,
                dt_tvd,
                c,
                g_left_tvd,
                g_right_tvd,
                size_tvd,
                &flux_func,
                &bc_tvd,
                Some(&recon_tvd),
            );
            t += dt_tvd;
        }
        dt_tvd = t_end - t;
        u_tvd = integrator_tvd.update(
            &u_tvd,
            dx_tvd,
            dt_tvd,
            c,
            g_left_tvd,
            g_right_tvd,
            size_tvd,
            &flux_func,
            &bc_tvd,
            Some(&recon_tvd),
        );

        for i in g_left_tvd..(size_tvd - g_right_tvd) {
            let x = mesh_tvd[i];
            let x_l = x - 0.5 * dx_tvd;
            let x_r = x + 0.5 * dx_tvd;
            let x_ana_l = periodic_wrap(x_l - c * t_end, x_begin, x_end);
            let x_ana_r = periodic_wrap(x_r - c * t_end, x_begin, x_end);

            u_exact_tvd[i] = 1.0 / (dx_tvd * 2.0 * PI) * (ana_int(x_ana_l) - ana_int(x_ana_r));
        }

        let err_tvd = calc_err(
            &u_tvd,
            &u_exact_tvd,
            size_tvd,
            g_left_tvd,
            g_right_tvd,
            dx_tvd,
        );

        //weno
        let mesh_weno = Mesh1d::new(x_begin, x_end, current_mesh_size, 2);
        let dx_weno = mesh_weno.get_dx();
        let size_weno = mesh_weno.get_length();
        let g_left_weno = mesh_weno.ghost_left();
        let g_right_weno = mesh_weno.ghost_right();

        let bc_weno = PerioidcBCWENO;
        let recon_weno = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };
        let integrator_weno = SspRk3IntegratorWENO;

        let mut u_weno = initialize_mesh(&mesh_weno, init_weno, &bc_weno);
        let mut u_exact_weno = u_weno.clone();

        let s_max = law.max_speed(&u_weno);
        let mut dt_weno = calc_time_step(dx_weno, cfl_weno, s_max);
        t = t_begin;

        while t < t_end {
            u_weno = integrator_weno.update(
                &u_weno,
                dx_weno,
                dt_weno,
                c,
                g_left_weno,
                g_right_weno,
                size_weno,
                &flux_func,
                &bc_weno,
                Some(&recon_weno),
            );
            t += dt_weno;
        }
        dt_weno = t_end - t;
        u_weno = integrator_weno.update(
            &u_weno,
            dx_weno,
            dt_weno,
            c,
            g_left_weno,
            g_right_weno,
            size_weno,
            &flux_func,
            &bc_weno,
            Some(&recon_weno),
        );

        for i in g_left_weno..(size_weno - g_right_weno) {
            let x = mesh_weno[i];
            let x_l = x - 0.5 * dx_weno;
            let x_r = x + 0.5 * dx_weno;
            let x_ana_l = periodic_wrap(x_l - c * t_end, x_begin, x_end);
            let x_ana_r = periodic_wrap(x_r - c * t_end, x_begin, x_end);

            u_exact_weno[i] = 1.0 / (dx_weno * 2.0 * PI) * (ana_int(x_ana_l) - ana_int(x_ana_r));
        }

        let err_weno = calc_err(
            &u_weno,
            &u_exact_weno,
            size_weno,
            g_left_weno,
            g_right_weno,
            dx_weno,
        );

        let row = RowData {
            mesh: current_mesh_size,
            err_fo,
            err_tvd,
            err_weno,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn convergence_burger<IS, IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_std: &IS,
    init_weno: &IW,
    cfl_fo: f64,
    cfl_tvd: f64,
    cfl_weno: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IS: InitialCondition,
    IW: InitialCondition,
{
    let mut wtr = Writer::from_path(csv_path)?;

    let base_points = 20;
    let mesh_sizes: Vec<usize> = (0..8).map(|i| base_points * 2_usize.pow(i)).collect();
    let num_meshes = mesh_sizes.len();

    let flux_func = RusanovFlux;
    let law = Burger;

    let mesh_size_exact = 4 * base_points * 2_usize.pow(7);
    let mesh_exact = Mesh1d::new(x_begin, x_end, mesh_size_exact, 2);
    let dx_exact = mesh_exact.get_dx();
    let size_exact = mesh_exact.get_length();
    let g_l_exact = mesh_exact.ghost_left();
    let g_r_exact = mesh_exact.ghost_right();

    let bc_exact = PerioidcBCWENO;
    let integrator_exact = SspRk3IntegratorWENO;
    let recon_exact = WENOReconstruction {
        gamma_l,
        gamma_c,
        gamma_r,
        p,
    };

    let mut u_exact = initialize_mesh(&mesh_exact, init_weno, &bc_exact);
    let mut dt_exact;

    let mut t = t_begin;
    println!("calculating exact solution");
    while t < t_end {
        let s_max = law.max_speed(&u_exact);
        dt_exact = calc_time_step(dx_exact, cfl_weno, s_max);

        u_exact = integrator_exact.update(
            &u_exact,
            dx_exact,
            dt_exact,
            c,
            g_l_exact,
            g_r_exact,
            size_exact,
            &flux_func,
            &bc_exact,
            Some(&recon_exact),
        );

        t += dt_exact;
    }
    dt_exact = t_end - t;
    u_exact = integrator_exact.update(
        &u_exact,
        dx_exact,
        dt_exact,
        c,
        g_l_exact,
        g_r_exact,
        size_exact,
        &flux_func,
        &bc_exact,
        Some(&recon_exact),
    );

    for i in 0..num_meshes {
        let curr_mesh_size = mesh_sizes[i];
        println!("current mesh size: {}", curr_mesh_size);

        //prepare fo
        let mesh_fo = Mesh1d::new(x_begin, x_end, curr_mesh_size, 1);
        let dx_fo = mesh_fo.get_dx();
        let size_fo = mesh_fo.get_length();
        let g_l_fo = mesh_fo.ghost_left();
        let g_r_fo = mesh_fo.ghost_right();

        let bc_fo = PeriodicBC;
        let integrator_fo = FoIntegrator;

        let mut u_fo = initialize_mesh(&mesh_fo, init_std, &bc_fo);
        let mut dt_fo;

        //prepare tvd
        let mesh_tvd = Mesh1d::new(x_begin, x_end, curr_mesh_size, 1);
        let dx_tvd = mesh_tvd.get_dx();
        let size_tvd = mesh_tvd.get_length();
        let g_l_tvd = mesh_tvd.ghost_left();
        let g_r_tvd = mesh_tvd.ghost_right();

        let bc_tvd = PeriodicBC;
        let integrator_tvd = SspRk2Integrator;
        let recon_tvd = TVDReconstruction { limiter: McLimiter };

        let mut u_tvd = initialize_mesh(&mesh_tvd, init_std, &bc_tvd);
        let mut dt_tvd;

        //prepare weno
        let mesh_weno = Mesh1d::new(x_begin, x_end, curr_mesh_size, 2);
        let dx_weno = mesh_weno.get_dx();
        let size_weno = mesh_weno.get_length();
        let g_l_weno = mesh_weno.ghost_left();
        let g_r_weno = mesh_weno.ghost_right();

        let bc_weno = PerioidcBCWENO;
        let integrator_weno = SspRk3IntegratorWENO;
        let recon_weno = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };

        let mut u_weno = initialize_mesh(&mesh_weno, init_weno, &bc_weno);
        let mut dt_weno;

        //fo
        t = t_begin;
        while t < t_end {
            let s_max = law.max_speed(&u_fo);
            dt_fo = calc_time_step(dx_fo, cfl_fo, s_max);

            u_fo = integrator_fo.update(
                &u_fo, dx_fo, dt_fo, c, g_l_fo, g_r_fo, size_fo, &flux_func, &bc_fo, None,
            );

            t += dt_fo;
        }
        dt_fo = t_end - t;
        u_fo = integrator_fo.update(
            &u_fo, dx_fo, dt_fo, c, g_l_fo, g_r_fo, size_fo, &flux_func, &bc_fo, None,
        );

        //tvd
        t = t_begin;
        while t < t_end {
            let s_max = law.max_speed(&u_tvd);
            dt_tvd = calc_time_step(dx_tvd, cfl_tvd, s_max);

            u_tvd = integrator_tvd.update(
                &u_tvd,
                dx_tvd,
                dt_tvd,
                c,
                g_l_tvd,
                g_r_tvd,
                size_tvd,
                &flux_func,
                &bc_tvd,
                Some(&recon_tvd),
            );

            t += dt_tvd;
        }
        dt_tvd = t_end - t;
        u_tvd = integrator_tvd.update(
            &u_tvd,
            dx_tvd,
            dt_tvd,
            c,
            g_l_tvd,
            g_r_tvd,
            size_tvd,
            &flux_func,
            &bc_tvd,
            Some(&recon_tvd),
        );

        //weno
        t = t_begin;
        while t < t_end {
            let s_max = law.max_speed(&u_weno);
            dt_weno = calc_time_step(dx_weno, cfl_weno, s_max);

            u_weno = integrator_weno.update(
                &u_weno,
                dx_weno,
                dt_weno,
                c,
                g_l_weno,
                g_r_weno,
                size_weno,
                &flux_func,
                &bc_weno,
                Some(&recon_weno),
            );

            t += dt_weno;
        }
        dt_weno = t_end - t;
        u_weno = integrator_weno.update(
            &u_weno,
            dx_weno,
            dt_weno,
            c,
            g_l_weno,
            g_r_weno,
            size_weno,
            &flux_func,
            &bc_weno,
            Some(&recon_weno),
        );

        //error
        let err_fo = calc_err_burger(
            &u_fo, &u_exact, size_fo, size_exact, g_l_fo, g_r_fo, g_l_exact, g_r_exact, dx_fo,
        );

        let err_tvd = calc_err_burger(
            &u_tvd, &u_exact, size_tvd, size_exact, g_l_tvd, g_r_tvd, g_l_exact, g_r_exact, dx_tvd,
        );

        let err_weno = calc_err_burger(
            &u_weno, &u_exact, size_weno, size_exact, g_l_weno, g_r_weno, g_l_exact, g_r_exact,
            dx_weno,
        );

        let row = RowData {
            mesh: curr_mesh_size,
            err_fo,
            err_tvd,
            err_weno,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn convergence_lin_adv_pc<IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_weno: &IW,
    cfl_pc: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IW: InitialCondition,
{
    let ana_int = |x: f64| -> f64 { (2.0 * PI * x).cos() };

    let base_points = 20;
    let mesh_sizes: Vec<usize> = (0..8).map(|i| base_points * 2_usize.pow(i)).collect();
    let num_meshes = mesh_sizes.len();

    let law = LinearAdvection { c };
    let flux_func = LaxFriedrichFlux;
    let flux_ana = LinFlux;
    let ader_flux_space = ADERLinFluxSpace;
    let ader_flux_time = ADERLinFluxSpaceTime;

    let mut wtr = Writer::from_path(csv_path)?;

    for i in 0..num_meshes {
        let current_mesh_size = mesh_sizes[i];
        println!("current mesh: {}", current_mesh_size);

        //pc
        let mesh_pc = Mesh1d::new(x_begin, x_end, current_mesh_size, 2);
        let dx_pc = mesh_pc.get_dx();
        let size_pc = mesh_pc.get_length();
        let g_left_pc = mesh_pc.ghost_left();
        let g_right_pc = mesh_pc.ghost_right();

        let bc_pc = PerioidcBCWENO;
        let recon_pc = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };
        let integrator_pc = PredictorCorrector;

        let mut u_pc = initialize_mesh(&mesh_pc, init_weno, &bc_pc);
        let mut u_exact_pc = u_pc.clone();

        let s_max = law.max_speed(&u_pc);
        let mut dt_pc = calc_time_step(dx_pc, cfl_pc, s_max);
        let mut t = t_begin;

        while t < t_end {
            u_pc = integrator_pc.update(
                &u_pc,
                dx_pc,
                dt_pc,
                c,
                g_left_pc,
                g_right_pc,
                size_pc,
                &flux_ana,
                &ader_flux_space,
                &ader_flux_time,
                &flux_func,
                &bc_pc,
                Some(&recon_pc),
            );
            t += dt_pc;
        }
        dt_pc = t_end - t;
        u_pc = integrator_pc.update(
            &u_pc,
            dx_pc,
            dt_pc,
            c,
            g_left_pc,
            g_right_pc,
            size_pc,
            &flux_ana,
            &ader_flux_space,
            &ader_flux_time,
            &flux_func,
            &bc_pc,
            Some(&recon_pc),
        );

        for i in g_left_pc..(size_pc - g_right_pc) {
            let x = mesh_pc[i];
            let x_l = x - 0.5 * dx_pc;
            let x_r = x + 0.5 * dx_pc;
            let x_ana_l = periodic_wrap(x_l - c * t_end, x_begin, x_end);
            let x_ana_r = periodic_wrap(x_r - c * t_end, x_begin, x_end);

            u_exact_pc[i] = 1.0 / (dx_pc * 2.0 * PI) * (ana_int(x_ana_l) - ana_int(x_ana_r));
        }

        let err_pc = calc_err(&u_pc, &u_exact_pc, size_pc, g_left_pc, g_right_pc, dx_pc);

        let row = RowData1 {
            mesh: current_mesh_size,
            err: err_pc,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn convergence_burger_pc<IS, IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_weno: &IW,
    init_std: &IS,
    cfl_pc: f64,
    cfl_weno: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IS: InitialCondition,
    IW: InitialCondition,
{
    let mut wtr = Writer::from_path(csv_path)?;

    let base_points = 20;
    let mesh_sizes: Vec<usize> = (0..8).map(|i| base_points * 2_usize.pow(i)).collect();
    let num_meshes = mesh_sizes.len();

    let flux_func = RusanovFlux;
    let flux_ana = BurgerFlux;
    let ader_flux_space = ADERBurgerFluxSpace;
    let ader_flux_time = ADERBurgerFluxSpaceTime;
    let law = Burger;

    let mesh_size_exact = 4 * base_points * 2_usize.pow(7);
    let mesh_exact = Mesh1d::new(x_begin, x_end, mesh_size_exact, 2);
    let dx_exact = mesh_exact.get_dx();
    let size_exact = mesh_exact.get_length();
    let g_l_exact = mesh_exact.ghost_left();
    let g_r_exact = mesh_exact.ghost_right();

    let bc_exact = PerioidcBCWENO;
    let integrator_exact = SspRk3IntegratorWENO;
    let recon_exact = WENOReconstruction {
        gamma_l,
        gamma_c,
        gamma_r,
        p,
    };

    let mut u_exact = initialize_mesh(&mesh_exact, init_weno, &bc_exact);
    let mut dt_exact;

    let mut t = t_begin;
    println!("calculating exact solution");
    while t < t_end {
        let s_max = law.max_speed(&u_exact);
        dt_exact = calc_time_step(dx_exact, cfl_weno, s_max);

        u_exact = integrator_exact.update(
            &u_exact,
            dx_exact,
            dt_exact,
            c,
            g_l_exact,
            g_r_exact,
            size_exact,
            &flux_func,
            &bc_exact,
            Some(&recon_exact),
        );

        t += dt_exact;
    }
    dt_exact = t_end - t;
    u_exact = integrator_exact.update(
        &u_exact,
        dx_exact,
        dt_exact,
        c,
        g_l_exact,
        g_r_exact,
        size_exact,
        &flux_func,
        &bc_exact,
        Some(&recon_exact),
    );

    for i in 0..num_meshes {
        let curr_mesh_size = mesh_sizes[i];
        println!("current mesh size: {}", curr_mesh_size);

        //prepare pc
        let mesh_pc = Mesh1d::new(x_begin, x_end, curr_mesh_size, 2);
        let dx_pc = mesh_pc.get_dx();
        let size_pc = mesh_pc.get_length();
        let g_l_pc = mesh_pc.ghost_left();
        let g_r_pc = mesh_pc.ghost_right();

        let bc_pc = PerioidcBCWENO;
        let integrator_pc = PredictorCorrector;
        let recon_pc = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };

        let mut u_pc = initialize_mesh(&mesh_pc, init_std, &bc_pc);
        let mut dt_pc;

        //pc
        t = t_begin;
        while t < t_end {
            let s_max = law.max_speed(&u_pc);
            dt_pc = calc_time_step(dx_pc, cfl_pc, s_max);

            u_pc = integrator_pc.update(
                &u_pc,
                dx_pc,
                dt_pc,
                c,
                g_l_pc,
                g_r_pc,
                size_pc,
                &flux_ana,
                &ader_flux_space,
                &ader_flux_time,
                &flux_func,
                &bc_pc,
                Some(&recon_pc),
            );

            t += dt_pc;
        }
        dt_pc = t_end - t;
        u_pc = integrator_pc.update(
            &u_pc,
            dx_pc,
            dt_pc,
            c,
            g_l_pc,
            g_r_pc,
            size_pc,
            &flux_ana,
            &ader_flux_space,
            &ader_flux_time,
            &flux_func,
            &bc_pc,
            Some(&recon_pc),
        );

        //error
        println!("calculating error");

        let err_pc = calc_err_burger(
            &u_pc, &u_exact, size_pc, size_exact, g_l_pc, g_r_pc, g_l_exact, g_r_exact, dx_pc,
        );

        let row = RowData1 {
            mesh: curr_mesh_size,
            err: err_pc,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn convergence_lin_adv_mp_pc<IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_weno: &IW,
    cfl_pc: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IW: InitialCondition,
{
    let ana_int = |x: f64| -> f64 { (2.0 * PI * x).cos() };

    let base_points = 20;
    let mesh_sizes: Vec<usize> = (0..8).map(|i| base_points * 2_usize.pow(i)).collect();
    let num_meshes = mesh_sizes.len();

    let law = LinearAdvection { c };
    let flux_func = LaxFriedrichFlux;
    let flux_ana = LinFlux;

    let mut wtr = Writer::from_path(csv_path)?;

    for i in 0..num_meshes {
        let current_mesh_size = mesh_sizes[i];
        println!("current mesh: {}", current_mesh_size);

        //pc
        let mesh_pc = Mesh1d::new(x_begin, x_end, current_mesh_size, 2);
        let dx_pc = mesh_pc.get_dx();
        let size_pc = mesh_pc.get_length();
        let g_left_pc = mesh_pc.ghost_left();
        let g_right_pc = mesh_pc.ghost_right();

        let bc_pc = PerioidcBCWENO;
        let recon_pc = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };
        let integrator_pc = MPPredictorCorrector;

        let mut u_pc = initialize_mesh(&mesh_pc, init_weno, &bc_pc);
        let mut u_exact_pc = u_pc.clone();

        let s_max = law.max_speed(&u_pc);
        let min = u_pc.min();
        let max = u_pc.max();
        let mut dt_pc = calc_time_step(dx_pc, cfl_pc, s_max);
        let mut t = t_begin;

        while t < t_end {
            u_pc = integrator_pc.update(
                min,
                max,
                &u_pc,
                dx_pc,
                dt_pc,
                c,
                g_left_pc,
                g_right_pc,
                size_pc,
                &flux_ana,
                &flux_func,
                &bc_pc,
                Some(&recon_pc),
            );
            t += dt_pc;
        }
        dt_pc = t_end - t;
        u_pc = integrator_pc.update(
            min,
            max,
            &u_pc,
            dx_pc,
            dt_pc,
            c,
            g_left_pc,
            g_right_pc,
            size_pc,
            &flux_ana,
            &flux_func,
            &bc_pc,
            Some(&recon_pc),
        );

        for i in g_left_pc..(size_pc - g_right_pc) {
            let x = mesh_pc[i];
            let x_l = x - 0.5 * dx_pc;
            let x_r = x + 0.5 * dx_pc;
            let x_ana_l = periodic_wrap(x_l - c * t_end, x_begin, x_end);
            let x_ana_r = periodic_wrap(x_r - c * t_end, x_begin, x_end);

            u_exact_pc[i] = 1.0 / (dx_pc * 2.0 * PI) * (ana_int(x_ana_l) - ana_int(x_ana_r));
        }

        let err_pc = calc_err(&u_pc, &u_exact_pc, size_pc, g_left_pc, g_right_pc, dx_pc);

        let row = RowData1 {
            mesh: current_mesh_size,
            err: err_pc,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn convergence_burger_mp_pc<IS, IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_weno: &IW,
    init_std: &IS,
    cfl_pc: f64,
    cfl_weno: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IS: InitialCondition,
    IW: InitialCondition,
{
    let mut wtr = Writer::from_path(csv_path)?;

    let base_points = 20;
    let mesh_sizes: Vec<usize> = (0..8).map(|i| base_points * 2_usize.pow(i)).collect();
    let num_meshes = mesh_sizes.len();

    let flux_func = RusanovFlux;
    let flux_ana = BurgerFlux;
    let law = Burger;

    let mesh_size_exact = 4 * base_points * 2_usize.pow(7);
    let mesh_exact = Mesh1d::new(x_begin, x_end, mesh_size_exact, 2);
    let dx_exact = mesh_exact.get_dx();
    let size_exact = mesh_exact.get_length();
    let g_l_exact = mesh_exact.ghost_left();
    let g_r_exact = mesh_exact.ghost_right();

    let bc_exact = PerioidcBCWENO;
    let integrator_exact = SspRk3IntegratorWENO;
    let recon_exact = WENOReconstruction {
        gamma_l,
        gamma_c,
        gamma_r,
        p,
    };

    let mut u_exact = initialize_mesh(&mesh_exact, init_weno, &bc_exact);
    let mut dt_exact;

    let mut t = t_begin;
    println!("calculating exact solution");
    while t < t_end {
        let s_max = law.max_speed(&u_exact);
        dt_exact = calc_time_step(dx_exact, cfl_weno, s_max);

        u_exact = integrator_exact.update(
            &u_exact,
            dx_exact,
            dt_exact,
            c,
            g_l_exact,
            g_r_exact,
            size_exact,
            &flux_func,
            &bc_exact,
            Some(&recon_exact),
        );

        t += dt_exact;
    }
    dt_exact = t_end - t;
    u_exact = integrator_exact.update(
        &u_exact,
        dx_exact,
        dt_exact,
        c,
        g_l_exact,
        g_r_exact,
        size_exact,
        &flux_func,
        &bc_exact,
        Some(&recon_exact),
    );

    for i in 0..num_meshes {
        let curr_mesh_size = mesh_sizes[i];
        println!("current mesh size: {}", curr_mesh_size);

        //prepare pc
        let mesh_pc = Mesh1d::new(x_begin, x_end, curr_mesh_size, 2);
        let dx_pc = mesh_pc.get_dx();
        let size_pc = mesh_pc.get_length();
        let g_l_pc = mesh_pc.ghost_left();
        let g_r_pc = mesh_pc.ghost_right();

        let bc_pc = PerioidcBCWENO;
        let integrator_pc = MPPredictorCorrector;
        let recon_pc = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };

        let mut u_pc = initialize_mesh(&mesh_pc, init_std, &bc_pc);
        let mut dt_pc;
        let min = u_pc.min();
        let max = u_pc.max();

        //pc
        t = t_begin;
        while t < t_end {
            let s_max = law.max_speed(&u_pc);
            dt_pc = calc_time_step(dx_pc, cfl_pc, s_max);

            u_pc = integrator_pc.update(
                min,
                max,
                &u_pc,
                dx_pc,
                dt_pc,
                c,
                g_l_pc,
                g_r_pc,
                size_pc,
                &flux_ana,
                &flux_func,
                &bc_pc,
                Some(&recon_pc),
            );

            t += dt_pc;
        }
        dt_pc = t_end - t;
        u_pc = integrator_pc.update(
            min,
            max,
            &u_pc,
            dx_pc,
            dt_pc,
            c,
            g_l_pc,
            g_r_pc,
            size_pc,
            &flux_ana,
            &flux_func,
            &bc_pc,
            Some(&recon_pc),
        );

        //error
        println!("calculating error");

        let err_pc = calc_err_burger(
            &u_pc, &u_exact, size_pc, size_exact, g_l_pc, g_r_pc, g_l_exact, g_r_exact, dx_pc,
        );

        let row = RowData1 {
            mesh: curr_mesh_size,
            err: err_pc,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn convergence_lin_adv_ader3<IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_weno: &IW,
    cfl_pc: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IW: InitialCondition,
{
    let ana_int = |x: f64| -> f64 { (2.0 * PI * x).cos() };

    let base_points = 50;
    let mesh_sizes: Vec<usize> = (1..=8).map(|i| base_points * i).collect();
    let num_meshes = mesh_sizes.len();

    let law = LinearAdvection { c };
    let flux_func = LaxFriedrichFlux;
    let flux_ana = LinFlux;
    let ader_flux_space = ADERLinFluxSpace;
    let ader_flux_time = ADERLinFluxSpaceTime;

    let mut wtr = Writer::from_path(csv_path)?;

    for i in 0..num_meshes {
        let current_mesh_size = mesh_sizes[i];
        println!("current mesh: {}", current_mesh_size);

        //pc
        let mesh_pc = Mesh1d::new(x_begin, x_end, current_mesh_size, 2);
        let dx_pc = mesh_pc.get_dx();
        let size_pc = mesh_pc.get_length();
        let g_left_pc = mesh_pc.ghost_left();
        let g_right_pc = mesh_pc.ghost_right();

        let bc_pc = PerioidcBCWENO;
        let recon_pc = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };
        let integrator_pc = ADER1DThirdOrder;

        let mut u_pc = initialize_mesh(&mesh_pc, init_weno, &bc_pc);
        let mut u_exact_pc = u_pc.clone();

        let s_max = law.max_speed(&u_pc);
        let mut dt_pc = calc_time_step(dx_pc, cfl_pc, s_max);
        let mut t = t_begin;

        while t < t_end {
            u_pc = integrator_pc.update(
                &u_pc,
                dx_pc,
                dt_pc,
                c,
                g_left_pc,
                g_right_pc,
                size_pc,
                &flux_ana,
                &ader_flux_space,
                &ader_flux_time,
                &flux_func,
                &bc_pc,
                Some(&recon_pc),
            );
            t += dt_pc;
        }
        dt_pc = t_end - t;
        u_pc = integrator_pc.update(
            &u_pc,
            dx_pc,
            dt_pc,
            c,
            g_left_pc,
            g_right_pc,
            size_pc,
            &flux_ana,
            &ader_flux_space,
            &ader_flux_time,
            &flux_func,
            &bc_pc,
            Some(&recon_pc),
        );

        for i in g_left_pc..(size_pc - g_right_pc) {
            let x = mesh_pc[i];
            let x_l = x - 0.5 * dx_pc;
            let x_r = x + 0.5 * dx_pc;
            let x_ana_l = periodic_wrap(x_l - c * t_end, x_begin, x_end);
            let x_ana_r = periodic_wrap(x_r - c * t_end, x_begin, x_end);

            u_exact_pc[i] = 1.0 / (dx_pc * 2.0 * PI) * (ana_int(x_ana_l) - ana_int(x_ana_r));
        }

        let err_pc = calc_err(&u_pc, &u_exact_pc, size_pc, g_left_pc, g_right_pc, dx_pc);

        let row = RowData1 {
            mesh: current_mesh_size,
            err: err_pc,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}

pub fn convergence_burger_ader3<IS, IW>(
    x_begin: f64,
    x_end: f64,
    t_begin: f64,
    t_end: f64,
    c: f64,
    gamma_l: f64,
    gamma_c: f64,
    gamma_r: f64,
    p: i32,
    init_weno: &IW,
    init_std: &IS,
    cfl_pc: f64,
    cfl_weno: f64,
    csv_path: &str,
) -> Result<(), Box<dyn Error>>
where
    IS: InitialCondition,
    IW: InitialCondition,
{
    let mut wtr = Writer::from_path(csv_path)?;

    let base_points = 50;
    let mesh_sizes: Vec<usize> = (1..=8).map(|i| base_points * i).collect();
    let num_meshes = mesh_sizes.len();

    let flux_func = RusanovFlux;
    let flux_ana = BurgerFlux;
    let ader_flux_space = ADERBurgerFluxSpace;
    let ader_flux_time = ADERBurgerFluxSpaceTime;
    let law = Burger;

    let mesh_size_exact = 4 * base_points * 8;
    let mesh_exact = Mesh1d::new(x_begin, x_end, mesh_size_exact, 2);
    let dx_exact = mesh_exact.get_dx();
    let size_exact = mesh_exact.get_length();
    let g_l_exact = mesh_exact.ghost_left();
    let g_r_exact = mesh_exact.ghost_right();

    let bc_exact = PerioidcBCWENO;
    let integrator_exact = SspRk3IntegratorWENO;
    let recon_exact = WENOReconstruction {
        gamma_l,
        gamma_c,
        gamma_r,
        p,
    };

    let mut u_exact = initialize_mesh(&mesh_exact, init_weno, &bc_exact);
    let mut dt_exact;

    let mut t = t_begin;
    println!("calculating exact solution");
    while t < t_end {
        let s_max = law.max_speed(&u_exact);
        dt_exact = calc_time_step(dx_exact, cfl_weno, s_max);

        u_exact = integrator_exact.update(
            &u_exact,
            dx_exact,
            dt_exact,
            c,
            g_l_exact,
            g_r_exact,
            size_exact,
            &flux_func,
            &bc_exact,
            Some(&recon_exact),
        );

        t += dt_exact;
    }
    dt_exact = t_end - t;
    u_exact = integrator_exact.update(
        &u_exact,
        dx_exact,
        dt_exact,
        c,
        g_l_exact,
        g_r_exact,
        size_exact,
        &flux_func,
        &bc_exact,
        Some(&recon_exact),
    );

    for i in 0..num_meshes {
        let curr_mesh_size = mesh_sizes[i];
        println!("current mesh size: {}", curr_mesh_size);

        //prepare pc
        let mesh_pc = Mesh1d::new(x_begin, x_end, curr_mesh_size, 2);
        let dx_pc = mesh_pc.get_dx();
        let size_pc = mesh_pc.get_length();
        let g_l_pc = mesh_pc.ghost_left();
        let g_r_pc = mesh_pc.ghost_right();

        let bc_pc = PerioidcBCWENO;
        let integrator_pc = ADER1DThirdOrder;
        let recon_pc = WENOReconstruction {
            gamma_l,
            gamma_c,
            gamma_r,
            p,
        };

        let mut u_pc = initialize_mesh(&mesh_pc, init_std, &bc_pc);
        let mut dt_pc;

        //pc
        t = t_begin;
        while t < t_end {
            let s_max = law.max_speed(&u_pc);
            dt_pc = calc_time_step(dx_pc, cfl_pc, s_max);

            u_pc = integrator_pc.update(
                &u_pc,
                dx_pc,
                dt_pc,
                c,
                g_l_pc,
                g_r_pc,
                size_pc,
                &flux_ana,
                &ader_flux_space,
                &ader_flux_time,
                &flux_func,
                &bc_pc,
                Some(&recon_pc),
            );

            t += dt_pc;
        }
        dt_pc = t_end - t;
        u_pc = integrator_pc.update(
            &u_pc,
            dx_pc,
            dt_pc,
            c,
            g_l_pc,
            g_r_pc,
            size_pc,
            &flux_ana,
            &ader_flux_space,
            &ader_flux_time,
            &flux_func,
            &bc_pc,
            Some(&recon_pc),
        );

        //error
        println!("calculating error");

        let err_pc = calc_err_burger(
            &u_pc, &u_exact, size_pc, size_exact, g_l_pc, g_r_pc, g_l_exact, g_r_exact, dx_pc,
        );

        let row = RowData1 {
            mesh: curr_mesh_size,
            err: err_pc,
        };
        wtr.serialize(row)?;
        wtr.flush()?;
    }

    Ok(())
}
