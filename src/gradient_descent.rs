use itertools::FoldWhile::{Continue, Done};
use itertools::Itertools;
use crate::data::create_datasets::Datum;

pub fn update_weights(weights: Vec<f64>, gradients: Vec<f64>, learning_rate: f64) -> Vec<f64>{
    // pair up weights and gradients with zip()
    let zipped = weights.iter().zip(gradients.iter());
    // return a new vector with the result of the function passed into map() at each index
    // map is lazy so we must call collect() to get the result, otherwise it's just an iterator
    zipped.map(|(weight, gradient)| weight - (gradient * learning_rate)).collect()
}


pub fn gradient_descent(data: &Vec<Datum>, weights: Vec<f64>,
                        loss_function: fn(data: &Vec<Datum>, weights: &Vec<f64>) -> f64,
                        calculate_gradients: fn(data: &Vec<Datum>, weights: &Vec<f64>) -> Vec<f64>,
                        learning_rate: f64, min_difference: f64, max_iterations: i32) -> (Vec<f64>, f64, f64) {
    // a..b creates a range from a to b, into iter makes an iterator that counts from a to b
    // fold_while is like fold except it breaks if you return the Done enum
    return (0..max_iterations).into_iter().fold_while((weights, 0.0, 0.0), |(new_weights, loss, _loss_difference), _| {
        let gradients = calculate_gradients(&data, &new_weights);
        let new_weights = update_weights(new_weights, gradients, learning_rate);
        let new_loss = loss_function(&data, &new_weights);
        let loss_difference = (loss - new_loss).abs();
        if loss_difference < min_difference { Done((new_weights, new_loss, loss_difference)) }
        else { Continue((new_weights, new_loss, loss_difference)) }
        // into inner unpacks the result
    }).into_inner();

}



#[cfg(test)]
mod tests {
    use super::super::parabola_model;
    use super::*;
    #[test]
    fn update_weights_test() {
        let t1 = update_weights(vec![1.0], vec![-0.25], 2.0);
        assert_eq!(1.5, t1[0]);
        let t2 = update_weights(vec![1.0, 3.0], vec![-0.25, 0.25], 2.0);
        assert_eq!(2.5, t2[1]);
    }

    #[test]
    fn gradient_descent_test() {
        let initial_weights = vec![3.0, 7.0];
        let data = vec![Datum{ features: vec![], label: 0 }];
        let (new_weights, _num_iterations, loss) =
            gradient_descent(&data, initial_weights,
                             parabola_model::loss_function,
                             parabola_model::calculate_gradients,
                             0.1, 0.001, 100);
        assert!(0.01 > loss);
        assert!(1.1 > new_weights[0] && new_weights[0] > 0.9);
        assert!(2.1 > new_weights[1] && new_weights[1] > 1.9)
    }
}