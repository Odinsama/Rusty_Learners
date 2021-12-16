mod math_utils;
mod parabola_model;
mod gradient_descent;
mod k_means_clustering;
mod data;
mod binomial_model;
mod stochastic_gradient_descent;
mod linear_regression_model;

fn main() {
    let vec1 = vec![1.0,3.0,4.0,8.0];
    let vec2 = vec![1.0,3.0,4.0,5.0];
    println!("Hello, world!{}", math::dot(&vec1, &vec2));
}
