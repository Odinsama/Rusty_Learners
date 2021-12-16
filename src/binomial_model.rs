use crate::data::create_datasets::Datum;

pub fn loss_function(data: &Vec<Datum>, weights: &Vec<f64>) -> f64 {
    // if we don't use chunks the result will get too small and become 0.0
    let batches = data.chunks(100);
    let batch_sum = batches.fold(0.0, | acc, chunk| {
        let sum = chunk.iter().fold(1.0, | acc: f64, datum| {
            let y = datum.label;
            // we don't need features since the only feature of a coin toss is the result
            let bias = weights[0];
            if y == 1 { acc * bias } else { acc * (1.0 - bias) }

        });
        acc + sum
    });

    return -(batch_sum).ln()
}

// pub fn calculate_gradient(data: &Vec<Datum>, weights: &Vec<f64>){
//
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::create_datasets::create_coin_dataset;
    use std::f64::{INFINITY};

    #[test]
    fn loss_function_test() {
        let data = create_coin_dataset(0.7743, 200);
        let w00 = vec![1.0];
        let w01 = vec![0.001];
        let result00 = loss_function(&data, &w00);
        let result01 = loss_function(&data, &w01);
        // since weight is 1.0 and it's wrong, we should get infinite loss because
        // there are in fact 0.0 labels in the data
        // weights that are too weak will also result in infinity, because the loss decreases
        // so much it goes out of bounds and become 0.0 which ln turns into infinity
        assert_eq!(result00, INFINITY);
        assert_eq!(result01, INFINITY);
        let w1 = vec![0.01];
        let result1 = loss_function(&data, &w1);
        let w2 = vec![0.5];
        let result2 = loss_function(&data, &w2);
        let w3 = vec![0.2];
        let result3 = loss_function(&data, &w3);
        // the true probability is 0.77 so we should expect to have more loss
        // the further we stray from that
        println!("result1 is: {}", result1);
        println!("result2 is: {}", result2);
        println!("result3 is: {}", result3);


        assert!(result2 < result3);
        assert!(result1 > result3);
        let w4 = vec![0.9];
        let result4 = loss_function(&data, &w4);
        let w5 = vec![0.7743];
        let result5 = loss_function(&data, &w5);

        println!("result4 is: {}", result4);
        println!("result5 is: {}", result5);
        // result 4 should be the closest yet, and w5 should be closest of all
        assert!((-1) > (-2));
        assert!(result4 < result3);
        assert!(result4 > result5);


        // same tests different p
        let data = create_coin_dataset(0.2743, 200);
        let w00 = vec![1.0];
        let w01 = vec![0.999];
        let result00 = loss_function(&data, &w00);
        let result01 = loss_function(&data, &w01);

        assert_eq!(result00, INFINITY);
        assert_eq!(result01, INFINITY);
        let w1 = vec![0.01];
        let result1 = loss_function(&data, &w1);
        let w2 = vec![0.5];
        let result2 = loss_function(&data, &w2);
        let w3 = vec![0.2];
        let result3 = loss_function(&data, &w3);

        println!("result1 is: {}", result1);
        println!("result2 is: {}", result2);
        println!("result3 is: {}", result3);


        assert!(result2 > result3);
        assert!(result1 > result3);
        let w4 = vec![0.9];
        let result4 = loss_function(&data, &w4);
        let w5 = vec![0.7743];
        let result5 = loss_function(&data, &w5);

        println!("result4 is: {}", result4);
        println!("result5 is: {}", result5);

        assert!((-1) > (-2));
        assert!(result4 > result3);
        assert!(result4 > result5);

        // same tests different p
        let data = create_coin_dataset(0.5, 200);
        let w00 = vec![1.0];
        let result00 = loss_function(&data, &w00);

        assert_eq!(result00, INFINITY);
        let w1 = vec![0.01];
        let result1 = loss_function(&data, &w1);
        let w2 = vec![0.5];
        let result2 = loss_function(&data, &w2);
        let w3 = vec![0.2];
        let result3 = loss_function(&data, &w3);

        println!("result1 is: {}", result1);
        println!("result2 is: {}", result2);
        println!("result3 is: {}", result3);


        assert!(result2 < result3);
        assert!(result1 > result3);
        let w4 = vec![0.9];
        let result4 = loss_function(&data, &w4);
        let w5 = vec![0.7743];
        let result5 = loss_function(&data, &w5);

        println!("result4 is: {}", result4);
        println!("result5 is: {}", result5);

        assert!((-1) > (-2));
        assert!(result4 > result3);
        assert!(result4 > result5);

    }
}