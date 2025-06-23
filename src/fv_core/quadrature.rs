extern crate nalgebra as na;

use na::SymmetricEigen;

fn golub_welsch(n: usize) -> (na::DVector<f64>, na::DVector<f64>) {
    let mut m: na::DMatrix<f64> = na::DMatrix::zeros(n, n);
    for i in 1..n {
        let b = (i as f64) / ((4 * i * i - 1) as f64).sqrt();
        m[(i, i - 1)] = b;
        m[(i - 1, i)] = b;
    }
    let eigen = SymmetricEigen::new(m);
    let nodes = eigen.eigenvalues;
    let eigen_vectors = eigen.eigenvectors;

    let weights_vec: Vec<f64> = eigen_vectors
        .column_iter()
        .map(|col| 2.0 * col[0] * col[0])
        .collect();
    let weights = na::DVector::from_vec(weights_vec);

    (nodes, weights)
}

pub fn gauss_legendre<F>(a: f64, b: f64, n: usize, integrand: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = golub_welsch(n);
    let mut integral_value = 0.0;
    let size = nodes.len();

    let weights_prefactor = 0.5 * (b - a);

    for i in 0..size {
        let x = 0.5 * (a + b) + 0.5 * (b - a) * nodes[i];
        integral_value += weights_prefactor * weights[i] * integrand(x);
    }
    integral_value
}
