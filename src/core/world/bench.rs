#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod load_bench {
    use std::{collections::HashMap, time::Duration, time::Instant};

    use glam::IVec3;

    use crate::core::{
        render::region::pack::{RegionData, pack_region},
        world::{World, grid::region_index_of, snapshot::MicroChunkSnapshot, snapshot::emit_snapshots},
    };

    #[test]
    #[ignore = "bench: cargo test --release load_pipeline_timings -- --ignored --nocapture"]
    fn load_pipeline_timings() {
        let path = std::env::var("ATLAS_BENCH_VOX")
            .unwrap_or_else(|_| "assets/church.vox".to_string());

        let start = Instant::now();
        let data = dot_vox::load(&path).unwrap();
        let parse = start.elapsed();

        let start = Instant::now();
        let (world, clipped) = World::new_clipped(&data);
        let world_new = start.elapsed();

        let start = Instant::now();
        let snapshots = emit_snapshots(&world).unwrap();
        let emit = start.elapsed();

        let start = Instant::now();
        let mut by_region: HashMap<IVec3, Vec<&MicroChunkSnapshot>> = HashMap::new();
        for snapshot in &snapshots {
            by_region
                .entry(region_index_of(snapshot.global_coords))
                .or_default()
                .push(snapshot);
        }

        let packed: Vec<RegionData> = by_region
            .into_iter()
            .map(|(region_index, region_snapshots)| {
                pack_region(region_index, &region_snapshots).unwrap()
            })
            .collect();
        let pack = start.elapsed();

        let total = parse + world_new + emit + pack;

        println!("path            {path}");
        println!("voxels          {}", world.voxel_count());
        println!("clipped         {clipped}");
        println!("micro chunks    {}", snapshots.len());
        println!("regions         {}", packed.len());
        println!("parse           {parse:10.3?}");
        println!("world_new       {world_new:10.3?}");
        println!("emit_snapshots  {emit:10.3?}");
        println!("pack            {pack:10.3?}");
        println!("total           {total:10.3?}");

        let budgets = [
            ("world_new", world_new, Duration::from_millis(600)),
            ("emit_snapshots", emit, Duration::from_millis(700)),
            ("pack", pack, Duration::from_millis(1000)),
            ("total", total, Duration::from_secs(3)),
        ];

        for (stage, elapsed, budget) in budgets {
            assert!(
                elapsed <= budget,
                "{stage} took {elapsed:.3?}, budget {budget:.3?} (regression in load pipeline)"
            );
        }
    }
}
