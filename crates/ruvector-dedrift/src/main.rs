//! Demo binary: runs the full DEDRIFT harness on synthetic drifting data and
//! prints a table of recall@10 / rebalance cost across five policies.

use ruvector_dedrift::{
    dedrift::{apply, Policy, PolicyConfig, PolicyReport},
    drift_sim::DriftWorld,
    recall_at_k, time_ms, Ivf, SmallRng,
};

const DIM: usize = 16;
const N_MODES: usize = 48;
const N_LISTS: usize = 16;
const N_INIT: usize = 12_000;
const N_TIMESTEPS: usize = 10;
const N_PER_STEP: usize = 1_500;
const N_QUERIES: usize = 200;
const K: usize = 10;
const NPROBE: usize = 1;

fn world_high_drift(dim: usize, n_modes: usize, seed: u64) -> DriftWorld {
    let mut w = DriftWorld::new(dim, n_modes, seed);
    w.mode_translation = 1.2;
    w.weight_rotation = 2.5;
    w.mode_sigma = 0.35;
    w
}

fn build_initial(world: &DriftWorld) -> (Ivf, Vec<Vec<f32>>) {
    let mut rng = SmallRng::new(42);
    let init = world.batch(N_INIT, 0.0, &mut rng);
    let mut ivf = Ivf::new(world.dim, N_LISTS);
    ivf.train(&init, 8, 101);
    for v in &init {
        ivf.add(v);
    }
    (ivf, init)
}

#[derive(Default)]
struct StepStat {
    recall: f32,
    search_ms: f64,
    maint_ms: f64,
    maint_splits: usize,
    maint_lazy: usize,
}

fn run_policy(world: &DriftWorld, policy: Policy) -> Vec<StepStat> {
    let (mut ivf, _) = build_initial(world);
    let mut rng = SmallRng::new(0xCAFE);
    let cfg = PolicyConfig::default();
    let mut stats: Vec<StepStat> = Vec::with_capacity(N_TIMESTEPS);

    for step in 1..=N_TIMESTEPS {
        let t = step as f32;

        // Insert a fresh batch of drifted vectors.
        let batch = world.batch(N_PER_STEP, t, &mut rng);
        for v in &batch {
            ivf.add(v);
        }

        // Apply maintenance.
        let report: PolicyReport = apply(&mut ivf, policy, &cfg);

        // Evaluate with queries drawn from the *current* distribution.
        let queries = world.batch(N_QUERIES, t, &mut rng);
        let mut recall_sum = 0.0f32;
        let (_, search_ms) = time_ms(|| {
            for q in &queries {
                let pred = ivf.search(q, K, NPROBE);
                let truth = ivf.brute(q, K);
                recall_sum += recall_at_k(&pred, &truth);
            }
        });

        stats.push(StepStat {
            recall: recall_sum / N_QUERIES as f32,
            search_ms: search_ms / N_QUERIES as f64,
            maint_ms: report.elapsed_ms,
            maint_splits: report.splits_applied,
            maint_lazy: report.lazy_recenters_applied,
        });
    }
    stats
}

fn mean<T: Copy + Into<f64>>(xs: &[T]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().map(|x| (*x).into()).sum::<f64>() / xs.len() as f64
}

fn summarize(label: &str, stats: &[StepStat]) {
    let recalls: Vec<f32> = stats.iter().map(|s| s.recall).collect();
    let final_recall = recalls.last().copied().unwrap_or(0.0);
    let mean_recall = mean(&recalls);
    let total_maint_ms: f64 = stats.iter().map(|s| s.maint_ms).sum();
    let mean_search_ms = mean(&stats.iter().map(|s| s.search_ms).collect::<Vec<_>>());
    let total_splits: usize = stats.iter().map(|s| s.maint_splits).sum();
    let total_lazy: usize = stats.iter().map(|s| s.maint_lazy).sum();
    println!(
        "{:<14} mean_recall@{K}={:.3}  final_recall@{K}={:.3}  search_ms/q={:.3}  maint_ms_total={:.1}  splits={}  lazy={}",
        label, mean_recall, final_recall, mean_search_ms, total_maint_ms, total_splits, total_lazy
    );
}

fn main() {
    println!(
        "DEDRIFT demo — dim={DIM} modes={N_MODES} lists={N_LISTS} init={N_INIT} steps={N_TIMESTEPS} per_step={N_PER_STEP} queries={N_QUERIES} k={K} nprobe={NPROBE}"
    );

    let world = world_high_drift(DIM, N_MODES, 31);

    let policies = [
        ("None", Policy::None),
        ("Split", Policy::Split),
        ("Lazy", Policy::Lazy),
        ("Hybrid", Policy::Hybrid),
        ("FullRebuild", Policy::FullRebuild),
    ];

    println!("\n--- per-policy aggregate ---");
    let mut all_stats: Vec<(&str, Vec<StepStat>)> = Vec::new();
    for (label, p) in policies {
        let stats = run_policy(&world, p);
        summarize(label, &stats);
        all_stats.push((label, stats));
    }

    // Pretty per-step recall table, columns are policies, rows are timesteps.
    println!("\n--- recall@{K} by timestep ---");
    print!("step  ");
    for (label, _) in &all_stats {
        print!("{:>13}", label);
    }
    println!();
    for step in 0..N_TIMESTEPS {
        print!("{:>4}  ", step + 1);
        for (_, stats) in &all_stats {
            print!("{:>13.3}", stats[step].recall);
        }
        println!();
    }

    println!("\n--- maintenance cost (ms) by timestep ---");
    print!("step  ");
    for (label, _) in &all_stats {
        print!("{:>13}", label);
    }
    println!();
    for step in 0..N_TIMESTEPS {
        print!("{:>4}  ", step + 1);
        for (_, stats) in &all_stats {
            print!("{:>13.2}", stats[step].maint_ms);
        }
        println!();
    }

    println!("\nDone.");
}
