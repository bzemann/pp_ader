extern crate nalgebra as na;

use crate::fv_core::mesh::Mesh;
use crate::fv_core::quadrature::gauss_legendre;

//traits
pub trait InitialCondition {
    fn compute(&self, mesh: &dyn Mesh) -> na::DVector<f64>;
}

pub trait BCEnforcer {
    fn enforce(&self, u: &mut na::DVector<f64>);
}

//structs for initial
pub struct StandardInit<F>
where
    F: Fn(f64) -> f64,
{
    pub f: F,
}

pub struct PPMInit<F>
where
    F: Fn(f64) -> f64,
{
    pub f: F,
}

pub struct WENOInit<F>
where
    F: Fn(f64) -> f64,
{
    pub f: F,
}

//structs for boundary
pub struct PeriodicBC;

pub struct PeriodicBCPPM;

pub struct PerioidcBCWENO;

//implementation for initial
impl<F> InitialCondition for StandardInit<F>
where
    F: Fn(f64) -> f64,
{
    fn compute(&self, mesh: &dyn Mesh) -> nalgebra::DVector<f64> {
        let dx = mesh.get_dx();
        let n = mesh.get_num_physical_points();
        let u_physical: Vec<f64> = mesh
            .grid_points()
            .iter()
            .skip(1)
            .take(n)
            .map(|&x| {
                let left = x - 0.5 * dx;
                let right = x + 0.5 * dx;
                gauss_legendre(left, right, 2, &self.f) / dx
            })
            .collect();

        let mut u_vec = Vec::with_capacity(n + 2);
        u_vec.push(0.0);
        u_vec.extend(u_physical);
        u_vec.push(0.0);
        na::DVector::from_vec(u_vec)
    }
}

impl<F> InitialCondition for PPMInit<F>
where
    F: Fn(f64) -> f64,
{
    fn compute(&self, mesh: &dyn Mesh) -> nalgebra::DVector<f64> {
        let dx = mesh.get_dx();
        let n = mesh.get_num_physical_points();
        let u_physical: Vec<f64> = mesh
            .grid_points()
            .iter()
            .skip(1)
            .take(n)
            .map(|&x| {
                let left = x - 0.5 * dx;
                let right = x + 0.5 * dx;
                gauss_legendre(left, right, 2, &self.f) / dx
            })
            .collect();

        let mut u_vec = Vec::with_capacity(n + 3);
        u_vec.push(0.0);
        u_vec.extend(u_physical);
        u_vec.push(0.0);
        u_vec.push(0.0);
        na::DVector::from_vec(u_vec)
    }
}

impl<F> InitialCondition for WENOInit<F>
where
    F: Fn(f64) -> f64,
{
    fn compute(&self, mesh: &dyn Mesh) -> nalgebra::DVector<f64> {
        let dx = mesh.get_dx();
        let n = mesh.get_num_physical_points();
        let u_physical: Vec<f64> = mesh
            .grid_points()
            .iter()
            .skip(2)
            .take(n)
            .map(|&x| {
                let left = x - 0.5 * dx;
                let right = x + 0.5 * dx;
                gauss_legendre(left, right, 2, &self.f) / dx
            })
            .collect();

        let mut u_vec = Vec::with_capacity(n + 4);
        u_vec.push(0.0);
        u_vec.push(0.0);
        u_vec.extend(u_physical);
        u_vec.push(0.0);
        u_vec.push(0.0);
        na::DVector::from_vec(u_vec)
    }
}

//implementation for boundary
impl BCEnforcer for PeriodicBC {
    fn enforce(&self, u: &mut nalgebra::DVector<f64>) {
        let length = u.len();
        if length < 2 {
            panic!("Not enough elements");
        }
        u[0] = u[length - 2];
        u[length - 1] = u[1];
    }
}

impl BCEnforcer for PeriodicBCPPM {
    fn enforce(&self, u: &mut nalgebra::DVector<f64>) {
        let length = u.len();
        if length < 5 {
            panic!("Not enough elements");
        }

        let n = length - 3;
        u[0] = u[n];

        u[n + 1] = u[1];
        u[n + 2] = u[2];
    }
}

impl BCEnforcer for PerioidcBCWENO {
    fn enforce(&self, u: &mut nalgebra::DVector<f64>) {
        let length = u.len();
        if length < 4 {
            panic!("Not enoug elements");
        }

        let n = length - 4;
        u[0] = u[n];
        u[1] = u[n + 1];
        u[n + 2] = u[2];
        u[n + 3] = u[3];
    }
}

pub fn initialize_mesh<I, B>(mesh: &dyn Mesh, init: &I, bc: &B) -> na::DVector<f64>
where
    I: InitialCondition,
    B: BCEnforcer,
{
    let mut u = init.compute(mesh);
    bc.enforce(&mut u);
    u
}
