use rand::Rng;
use std::fs::File;
use std::io::{BufReader, BufRead};

#[derive(Debug)]
pub struct Datum {
    pub(crate) features: Vec<f64>,
    pub(crate) label: i32
}

pub fn create_spambase_dataset() -> (Vec<Datum>, Vec<String>) {

    let path = "src/data/spambase.csv";
    let header_path = "src/data/spambase.header";
    // let mut output = File::create(path)?;
    // write!(output, "Rust\n💖\nFun")?;

    let header = File::open(header_path).unwrap();
    let input = File::open(path).unwrap();
    let buffered_header = BufReader::new(header);
    let buffered_data = BufReader::new(input);
    let line = buffered_header.lines().next().unwrap().unwrap();
    let feature_names: Vec<String> = line.split(",").map(|name| {String::from(name)}).collect();
    let data = buffered_data.lines().fold(vec![], |mut acc, line| {
        let line = line.unwrap();
        let mut features = vec![0.0; feature_names.len()];
        let mut label = 0;
        for (num, feat) in line.split(",").enumerate() {
            let feat_value: f64 = feat.parse().unwrap();
            if feature_names[num] == "label" { label = feat_value as i32; }
            features[num] = feat_value;
        }
        let d = Datum {
            features,
            label
        };
        acc.push(d);
        acc
    });
    (data, feature_names)
}

pub fn create_coin_dataset(p: f64, n: i32) -> Vec<Datum> {
    let mut rng = rand::thread_rng();
    let mut data: Vec<Datum> = vec![];
    for _ in 0..n {
        let num = rng.gen_range(0.0..1.0);
        data.push(Datum {
            // the 1.0 is bias
            features: vec![1.0],
            label: if num < p { 1 } else { 0 }
        });
    }
    return data
}

#[cfg(test)]
mod tests {
    use crate::data::create_datasets::create_spambase_dataset;

    #[test]
    fn loss_function_test() {
        let data = create_spambase_dataset();
        assert_eq!(data.0.len(), 4601)

    }
}