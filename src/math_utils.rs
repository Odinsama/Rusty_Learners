pub fn dot(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    // by zipping these equal length vectors we can iterate over them in pairs
    let zipped_vectors =  vec1.iter().zip(vec2.iter());
    // the product of each pair is added to the accumulator, returning our dot product
    zipped_vectors.fold(0.0, |acc, tuple| acc + (tuple.0 * tuple.1))
}
pub fn norm(vec1: &Vec<f64>) -> f64 {
    let sum_of_squares = vec1.into_iter().fold(0.0, |acc, n| acc + (n * n));
    sum_of_squares.sqrt()
}

pub fn mean(vec: &Vec<f64>) -> f64 {
    let sum = vec.iter().fold(0.0, |acc, n| acc + n);
    sum / (vec.len() as f64)
}

pub fn std_deviation(vec: Vec<f64>) -> f64 {
    let m = mean(&vec);
    let sum_deviations = vec.iter().fold(0.0, |acc, n| acc + ((n-m) * (n-m)));
    sum_deviations / (vec.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    #[test]
    fn dot_test() {
        assert_eq!(dot(&vec![1.0, 2.0, 3.0, 4.0], &vec![1.0, 2.0, 3.0, 4.0]), 30.0);
        assert_eq!(dot(&vec![4.0, 20.0, 2.0, 4.0], &vec![1.0, 2.0, 3.0, 8.0]), 82.0);
        assert_eq!(dot(&vec![4.6, 20.1, 2.7, 4.9], &vec![1.9, 2.5, 3.3, 8.5]), 109.55);
    }
    #[test]
    fn norm_test() {
        assert_eq!(norm(&vec![1.0, 2.0, 3.0, 4.0]), 5.477225575051661);
        assert_eq!(norm(&vec![4.0, 20.0, 2.0, 4.0]), 20.8806130178211);
        assert_eq!(norm(&vec![4.7, 20.2, 2.7, 4.9]), 21.480921767931655);
        assert_eq!(norm(&vec![SQRT_2, -SQRT_2]), 2.0);
    }
    #[test]
    fn mean_test() {
        assert_eq!(mean(&vec![1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(mean(&vec![4.0, 20.0, 2.0, 4.0]), 7.5);
        assert_eq!(mean(&vec![4.7, 20.2, 2.7, 4.9]), 8.125);
    }
    #[test]
    fn std_deviation_test() {

    }
}