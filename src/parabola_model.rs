use crate::data::create_datasets::Datum;

/** this function doesn't use feature values,
but we use the same signature for all loss functions as a style choice */
pub fn loss_function(_data: &Vec<Datum>, weights: &Vec<f64>) -> f64 {
    0.5 * ((weights[0] - 1.0) * (weights[0] - 1.0) + (weights[1] - 2.0) * (weights[1] - 2.0))
}
pub fn calculate_gradients(_data: &Vec<Datum>, weights: &Vec<f64>) -> Vec<f64> {
    // could have made this one line with no mut but I like it better this way
    let mut gradients: Vec<f64> = vec![];
    gradients.push(weights[0] - 1.0);
    gradients.push(weights[1] - 2.0);
    return gradients
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loss_function_test() {
        let t1 = loss_function(&vec![], &vec![1.0, 2.0]);
        assert_eq!(0.0, t1);
        let t2 = loss_function(&vec![], &vec![1.0, 1.0]);
        assert_eq!(0.5, t2);
        let t3 = loss_function(&vec![], &vec![1.0, 3.0]);
        assert_eq!(0.5, t3);
        let t4 = loss_function(&vec![], &vec![3.0, 1.0]);
        assert_eq!(2.5, t4);
    }
    #[test]
    fn calculate_gradients_test() {
        let t1 = calculate_gradients(&vec![], &vec![1.0, 2.0]);
        assert_eq!(0.0, t1[0]);
        assert_eq!(0.0, t1[1]);
        let t2 = calculate_gradients(&vec![], &vec![1.0, 1.0]);
        assert_eq!(0.0, t2[0]);
        assert_eq!(-1.0, t2[1]);
    }
}