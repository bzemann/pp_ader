use crate::fv_core::{
    condition::PerioidcBCWENO,
    flux::{ADERLinFluxSpace, ADERLinFluxSpaceTime, LaxFriedrichFlux, LinFlux},
    initial,
    mesh::Mesh1d,
    reconstruction::WENOReconstruction,
};

use crate::time_integrator::{
    ader::{solver_ader, ADER1DThirdOrder, PredictorCorrector},
    fvm_ssp_rk::LinearAdvection,
};

use crate::convergence::convergence_burger;

use std::f64::consts::PI;

fn analytical(x: f64) -> f64 {
    (2.0 * PI * x).sin()
}

pub fn ader_third_lin() -> Result<(), Box<dyn std::error::Error>> {
    let x_begin = 0.0;
    let x_end = 1.0;
    let t_begin = 0.0;
    let t_end = 0.5;
    let cfl = 0.001;
    let c = 1.0;

    let gamma_l = 1.0;
    let gamma_c = 50.0;
    let gamma_r = 1.0;
    let p = 4;

    let mesh = Mesh1d::new(x_begin, x_end, 400, 2);
    let law = LinearAdvection { c };
    let bc = PerioidcBCWENO;
    let flux_func = LaxFriedrichFlux;
    let ader_flux_space = ADERLinFluxSpace;
    let ader_flux_time = ADERLinFluxSpaceTime;
    let flux_ana = LinFlux;

    let recon = WENOReconstruction {
        gamma_l,
        gamma_c,
        gamma_r,
        p,
    };

    let integrator = ADER1DThirdOrder;
    let init = initial::sin_weno();

    let csv_path = "results/csv_files/plot_files/solution_lin_adv_ader_3.csv";

    solver_ader(
        &mesh,
        &init,
        &flux_func,
        &flux_ana,
        &ader_flux_space,
        &ader_flux_time,
        Some(recon),
        &bc,
        &law,
        integrator,
        c,
        cfl,
        t_begin,
        t_end,
        csv_path,
        x_begin,
        x_end,
        Some(&analytical),
    )?;

    Ok(())
}

pub fn pc_lin() -> Result<(), Box<dyn std::error::Error>> {
    let x_begin = 0.0;
    let x_end = 1.0;
    let t_begin = 0.0;
    let t_end = 0.5;
    let cfl = 0.9;
    let c = 1.0;

    let gamma_l = 1.0;
    let gamma_c = 50.0;
    let gamma_r = 1.0;
    let p = 4;

    let mesh = Mesh1d::new(x_begin, x_end, 400, 2);
    let law = LinearAdvection { c };
    let bc = PerioidcBCWENO;
    let flux_func = LaxFriedrichFlux;
    let ader_flux_space = ADERLinFluxSpace;
    let ader_flux_time = ADERLinFluxSpaceTime;
    let flux_ana = LinFlux;

    let recon = WENOReconstruction {
        gamma_l,
        gamma_c,
        gamma_r,
        p,
    };

    let integrator = PredictorCorrector;
    let init = initial::sin_weno();

    let csv_path = "results/csv_files/plot_files/solution_lin_adv_pc.csv";
    solver_ader(
        &mesh,
        &init,
        &flux_func,
        &flux_ana,
        &ader_flux_space,
        &ader_flux_time,
        Some(recon),
        &bc,
        &law,
        integrator,
        c,
        cfl,
        t_begin,
        t_end,
        csv_path,
        x_begin,
        x_end,
        Some(&analytical),
    )?;

    Ok(())
}

pub fn conv_burger() -> Result<(), Box<dyn std::error::Error>> {
    let x_begin = 0.0;
    let x_end = 1.0;
    let t_begin = 0.0;
    let t_end = 0.5;
    let cfl_fo = 0.9;
    let cfl_tvd = 0.8;
    let cfl_weno = 0.5;
    let c = 1.0;

    let gamma_l = 1.0;
    let gamma_c = 50.0;
    let gamma_r = 1.0;
    let p = 4;

    let init_std = initial::sin_standard();
    let init_weno = initial::sin_weno();

    let csv_path = "results/csv_files/convergence/conv_burger_ssp.csv";

    convergence_burger(
        x_begin, x_end, t_begin, t_end, c, gamma_l, gamma_c, gamma_r, p, &init_std, &init_weno,
        cfl_fo, cfl_tvd, cfl_weno, csv_path,
    )?;

    Ok(())
}
