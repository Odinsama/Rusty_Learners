use rand::prelude::*;
use rand_pcg::Pcg64;
use rand::distributions::Uniform;
use std::f64::MAX;


fn create_cluster_dataset() -> Vec<[f64;2]> {
    let n = 100;
    let clusters = [[-4.0, 4.0], [-4.0, -4.0], [4.0, 4.0], [4.0, -4.0], [0.0, 0.0]];
    let mut rng1 = Pcg64::seed_from_u64(1);
    let mut rng2 = Pcg64::seed_from_u64(2);
    let data: Vec<[f64;2]> = clusters.iter().fold(vec![], |mut acc, cluster| {
        let uniform_distribution1: Uniform<f64> = Uniform::new_inclusive(cluster[0] - 1.0, cluster[0] + 1.0);
        let uniform_distribution2: Uniform<f64> = Uniform::new_inclusive(cluster[1] - 1.0, cluster[1] + 1.0);
        for _ in 0..n {
            acc.push([uniform_distribution1.sample(&mut rng1), uniform_distribution2.sample(&mut rng2)]);
        }
        acc
    });
    return data
}
#[derive(Clone, Debug)]
struct Cluster {
    means: Vec<f64>,
    data: Vec<[f64;2]>
}
fn init_clusters(data: &Vec<[f64;2]>, k: i32) -> Vec<Cluster> {
    let mut rng = Pcg64::seed_from_u64(3);
    let uniform_distribution1: Uniform<usize> = Uniform::new(0, data.len());
    let mut clusters = vec![];
    for _ in 0..k {
        let m1 = data[uniform_distribution1.sample(&mut rng)][0];
        let m2 = data[uniform_distribution1.sample(&mut rng)][0];
        clusters.push(Cluster {
            means: vec![m1,m2],
            data: vec![]
        })
    }
    return clusters
}

fn assign_clusters(data: &Vec<[f64;2]>, clusters: &mut Vec<Cluster>) {
    // we wipe out previous assignment because
    // it's easier to assign from scratch than to move values around
    for n in 0..clusters.len() {
        clusters[n].data = vec![]
    }
    for n in 0..data.len() {
        let mut shortest_distance = MAX;
        let mut shortest_pos = 0;
        for k in 0..clusters.len() {
            let mut distance = 0.0;
            for m in 0..clusters[k].means.len() {
                let cluster = &clusters[k];
                let mean = cluster.means[m];
                let datum = data[n];
                let feature = datum[m];
                distance += (mean - feature).powi(2)
            }
            if distance < shortest_distance {
                shortest_distance = distance;
                shortest_pos = k;
            }
        }
        clusters[shortest_pos].data.push(data[n]);
    }
}

fn calculate_means(clusters: &mut Vec<Cluster>) {
    for mut cluster in clusters {
        let sums = cluster.data.iter().fold(vec![0.0; cluster.data[0].len()], |mut acc, datum| {
            for (i, d) in datum.iter().enumerate() { acc[i] += d }; acc
        });
        cluster.means = sums.iter().enumerate().fold(vec![0.0; sums.len()], |mut acc, (i, sum)| {
            acc[i] = sum / (cluster.data.len() as f64);
            return acc
        });
    }
}

fn k_means(data: Vec<[f64;2]>, k: i32, tau: f64) -> (Vec<Cluster>, Vec<f64>) {
    let mut clusters = init_clusters(&data, k);
    let mut distances = vec![];
    for _ in 0..100 {
        let old_clusters = clusters.clone();
        let old_means: Vec<&Vec<f64>> = old_clusters.iter().map(|cluster| { &cluster.means }).collect();
        assign_clusters(&data, &mut clusters);
        calculate_means(&mut clusters);
        let new_means: Vec<&Vec<f64>> = clusters.iter().map(|cluster| { &cluster.means }).collect();
        let zipped = old_means.iter().zip(new_means.iter());
        let distance = zipped.fold(0.0, |mut acc, (old_c, new_c)| {
            let zipped_squared = old_c.iter().zip(new_c.iter());
            let inner_distance = zipped_squared.fold(0.0, |mut acc, (old_m, new_m)| {
                acc += (old_m - new_m).abs();
                acc
            });
            acc += inner_distance;
            acc
        });
        distances.push(distance);
        if distance < tau {
            break;
        }
    }
    println!("finished! {:?}", distances);
    return (clusters, distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_cluster_test() {
        let data = create_cluster_dataset();
        assert_eq!(data.len(), 500);
    }
    #[test]
    fn init_clusters_test() {
        let data = create_cluster_dataset();
        let clusters = init_clusters(&data, 5);
        assert_eq!(clusters.len(), 5);
        let means = &clusters[0].means;
        assert_eq!(means.len(), 2);
        assert!(means[0].abs() < 10.0);
        assert!(means[1].abs() < 10.0);
    }
    #[test]
    fn calculate_means_test() {
        let dataset = create_cluster_dataset();
        let mut clusters = init_clusters(&dataset, 5);
        let mean1 = clusters[0].means[0];
        assign_clusters(&dataset, &mut clusters);
        calculate_means(&mut clusters);
        let mean2 = clusters[0].means[0];
        println!("{}",mean1);
        println!("{}",mean2);
        assert_ne!(mean1, mean2)
    }
    #[test]
    fn k_means_test() {
        let data = create_cluster_dataset();
        let (clusters, distances) = k_means(data, 3, 0.001);
        assert!(distances.len() > 3);
        assert_eq!(clusters.iter().fold(0, |mut acc, cluster| { acc += cluster.data.len(); acc }), 500)
    }
    
}

