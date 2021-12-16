use crate::data::create_datasets::Datum;
use crate::math::dot;

fn predict(datum: &Datum, weights: &Vec<f64>) -> f64 {
    dot(&datum.features, &weights)
}
fn loss_function(data: Vec<Datum>, weights: Vec<f64>) -> f64 {
    let sum = data.iter().fold(0.0, |acc, datum|{
        let p = predict(datum, &weights);
        let y = datum.label as f64;
        acc + (0.5 * (p - y).powi(2))
    });
    return sum / (data.len() as f64)
}
fn calculate_gradient(data: Vec<Datum>, weights: Vec<f64>) -> Vec<f64>{
    data.iter().fold(vec![], |mut acc, datum| {
        let p = predict(datum, &weights);
        datum.features.iter().for_each(|feat| {
            acc.push(feat * (p - datum.label as f64))
        });
        return acc
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::create_datasets::create_coin_dataset;

    #[test]
    fn predict_test() {
        let d = Datum{ features: vec![2.0], label: 0 };
        let weights = vec![3.0];
        assert_eq!(predict(&d, &weights), 6.0)
    }
    #[test]
    fn loss_function_test() {
        let data = vec![Datum{ features: vec![1.0], label: 0 }];
        let weights = vec![0.5];
        let loss = loss_function(data, weights);
        assert_eq!(loss, 0.125)
        

    }
    #[test]
    fn calculate_gradient_test() {
        let data = create_coin_dataset(0.1, 1000);
        let weights = vec![0.5];
        let loss = loss_function(data, weights);
        assert_eq!(loss, 0.125)


    }
}