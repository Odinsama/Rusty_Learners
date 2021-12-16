use crate::data::create_datasets::Datum;

fn init_stochastic_gradient_descent(
    calculate_gradient: fn(data: &Vec<Datum>, weights: &Vec<f64>) -> Vec<f64>,
    weights: &mut Vec<f64>,
    learning_rate: f64) -> Box<dyn FnMut(&Vec<Datum>) + '_> {
    let mut n: f64 = 1.0;
    Box::new(move |data|{
        let current_learning_rate = learning_rate * n.sqrt();
        let gradient = calculate_gradient(data, &weights);
        for i in 0..gradient.len() {
            weights[i] = weights[i] - (gradient[i] * current_learning_rate);
        }
        n = n + 1.0;
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parabola_model::calculate_gradients;

    #[test]
    fn init_stochastic_gradient_descent_test() {
        // with lr of 1 should behave exactly like the original
        let mut t1 = vec![1.0, 2.0];
        let mut sgd = init_stochastic_gradient_descent(
            calculate_gradients,
            &mut t1,
            1.0
        );
        // our calculate gradients have this signature but doesn't actually use data
        let data = &vec![];
        sgd(data);
        std::mem::drop(sgd);
        assert_eq!(0.0, t1[0]);
        assert_eq!(0.0, t1[1]);
        // let t2 = vec![1.0, 1.0];
        // let sgd = init_stochastic_gradient_descent(
        //     calculate_gradients,
        //     &t2,
        //     1.0
        // );
        // assert_eq!(0.0, t1[0]);
        // assert_eq!(-1.0, t1[1]);

    }
}