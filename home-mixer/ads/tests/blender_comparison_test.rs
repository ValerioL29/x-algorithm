use crate::ads::{
    partition_organic_blender, safe_gap_blender,
    util::{should_drop_handle, should_drop_keyword},
};
use crate::params::RESULT_SIZE;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo, BrandSafetyRiskLevel};

const UNSAFE_POST_RATE: f64 = 0.52;

const LOW_RISK_POST_RATE: f64 = 0.06;

const BSR_LOW_AD_RATE: f64 = 0.145;
const BSR_IAS_AD_RATE: f64 = 0.008;
const BSR_NO_RISK_AD_RATE: f64 = 0.029;

const KNOWN_HANDLE_POST_RATE: f64 = 0.10;
const AD_EXCLUDED_HANDLES_RATE: f64 = 0.10;
const AD_EXCLUDED_KEYWORDS_RATE: f64 = 0.10;

const HANDLE_POOL: &[i64] = &[1001, 1002, 1003, 1004, 1005];

const SCREEN_NAME_POOL: &[&str] = &["user_a", "user_b", "user_c", "user_d", "user_e"];

const KEYWORD_POOL: &[&str] = &["alpha", "bravo", "charlie", "delta", "echo"];

const KEYWORD_IN_TEXT_RATE: f64 = 0.10;

const SAFE_TEXT_POOL: &[&str] = &[
    "what a beautiful day outside",
    "just finished a great workout",
    "loving this new recipe I found",
    "anyone watching the game tonight",
    "happy birthday to my best friend",
];

const NUM_SCENARIOS: usize = 200;

const RNG_SEED: u64 = 412;

struct Scenario {
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
}

fn generate_scenarios(rng: &mut StdRng) -> Vec<Scenario> {
    (0..NUM_SCENARIOS)
        .map(|_| {
            let num_posts = rng.random_range(10..=35);
            let num_ads = rng.random_range(3..=15);

            let posts: Vec<ScoredPost> = (1..=num_posts)
                .map(|id| {
                    let roll: f64 = rng.random();
                    let verdict = if roll < UNSAFE_POST_RATE {
                        BrandSafetyVerdict::MediumRisk
                    } else if roll < UNSAFE_POST_RATE + LOW_RISK_POST_RATE {
                        BrandSafetyVerdict::LowRisk
                    } else {
                        BrandSafetyVerdict::Safe
                    };

                    let (author_id, screen_names) = if rng.random_bool(KNOWN_HANDLE_POST_RATE) {
                        let idx = rng.random_range(0..HANDLE_POOL.len());
                        let aid = HANDLE_POOL[idx] as u64;
                        let mut map = HashMap::new();
                        map.insert(aid, SCREEN_NAME_POOL[idx].to_string());
                        (aid, map)
                    } else {
                        (id as u64, HashMap::new())
                    };

                    let tweet_text = if rng.random_bool(KEYWORD_IN_TEXT_RATE) {
                        let kw = KEYWORD_POOL[rng.random_range(0..KEYWORD_POOL.len())];
                        format!("This post is about {kw} and other topics")
                    } else {
                        SAFE_TEXT_POOL[rng.random_range(0..SAFE_TEXT_POOL.len())].to_string()
                    };

                    ScoredPost {
                        tweet_id: id as u64,
                        author_id,
                        brand_safety_verdict: verdict.into(),
                        screen_names,
                        tweet_text,
                        ..Default::default()
                    }
                })
                .collect();

            let ads: Vec<AdIndexInfo> = (0..num_ads)
                .map(|i| {
                    let insert_position = i * 3 + 1;
                    let roll: f64 = rng.random();
                    let risk = if roll < BSR_LOW_AD_RATE {
                        BrandSafetyRiskLevel::BsrLow
                    } else if roll < BSR_LOW_AD_RATE + BSR_IAS_AD_RATE {
                        BrandSafetyRiskLevel::BsrIas
                    } else if roll < BSR_LOW_AD_RATE + BSR_IAS_AD_RATE + BSR_NO_RISK_AD_RATE {
                        BrandSafetyRiskLevel::BsrNoRisk
                    } else {
                        BrandSafetyRiskLevel::BsrNormal
                    };

                    let handles = if rng.random_bool(AD_EXCLUDED_HANDLES_RATE) {
                        let count = rng.random_range(1..=3);
                        (0..count)
                            .map(|_| HANDLE_POOL[rng.random_range(0..HANDLE_POOL.len())])
                            .collect()
                    } else {
                        vec![]
                    };

                    let keywords = if rng.random_bool(AD_EXCLUDED_KEYWORDS_RATE) {
                        let count = rng.random_range(1..=2);
                        (0..count)
                            .map(|_| {
                                KEYWORD_POOL[rng.random_range(0..KEYWORD_POOL.len())].to_string()
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                    AdIndexInfo {
                        post_id: 1000 + i as i64,
                        insert_position,
                        ad_adjacency_control: Some(AdAdjacencyControl {
                            brand_safety_risk: risk.into(),
                            keywords,
                            handles,
                        }),
                        ..Default::default()
                    }
                })
                .collect();

            Scenario { posts, ads }
        })
        .collect()
}

fn ad_count(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .count()
}

fn post_count(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
        .count()
}

fn ad_positions(items: &[FeedItem]) -> Vec<i32> {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .map(|i| i.position)
        .collect()
}

struct AdjacencyViolations {
    med_risk_adj: usize,
    bsr_low_low_risk_adj: usize,
    bsr_low_placed: usize,
    bsr_low_available: usize,
    handle_adj: usize,
    keyword_adj: usize,
}

fn count_adjacency_violations(
    items: &[FeedItem],
    input_ads: &[AdIndexInfo],
) -> AdjacencyViolations {
    let mut v = AdjacencyViolations {
        med_risk_adj: 0,
        bsr_low_low_risk_adj: 0,
        bsr_low_placed: 0,
        handle_adj: 0,
        keyword_adj: 0,
        bsr_low_available: input_ads
            .iter()
            .filter(|a| {
                a.ad_adjacency_control
                    .as_ref()
                    .map(|c| {
                        matches!(
                            c.brand_safety_risk(),
                            BrandSafetyRiskLevel::BsrLow | BrandSafetyRiskLevel::BsrIas
                        )
                    })
                    .unwrap_or(false)
            })
            .count(),
    };

    for (i, item) in items.iter().enumerate() {
        let ad = match &item.item {
            Some(feed_item::Item::Ad(ad)) => ad,
            _ => continue,
        };

        let is_sensitive = ad
            .ad_adjacency_control
            .as_ref()
            .map(|c| {
                matches!(
                    c.brand_safety_risk(),
                    BrandSafetyRiskLevel::BsrLow | BrandSafetyRiskLevel::BsrIas
                )
            })
            .unwrap_or(false);
        if is_sensitive {
            v.bsr_low_placed += 1;
        }

        let above_post = if i > 0 {
            match &items[i - 1].item {
                Some(feed_item::Item::Post(p)) => Some(p),
                _ => None,
            }
        } else {
            None
        };
        let below_post = if i + 1 < items.len() {
            match &items[i + 1].item {
                Some(feed_item::Item::Post(p)) => Some(p),
                _ => None,
            }
        } else {
            None
        };

        let above_verdict = above_post.map(|p| p.brand_safety_verdict());
        let below_verdict = below_post.map(|p| p.brand_safety_verdict());

        if above_verdict == Some(BrandSafetyVerdict::MediumRisk)
            || below_verdict == Some(BrandSafetyVerdict::MediumRisk)
        {
            v.med_risk_adj += 1;
        }

        if is_sensitive
            && (above_verdict == Some(BrandSafetyVerdict::LowRisk)
                || below_verdict == Some(BrandSafetyVerdict::LowRisk))
        {
            v.bsr_low_low_risk_adj += 1;
        }

        if should_drop_handle(ad, above_post, below_post) {
            v.handle_adj += 1;
        }

        if should_drop_keyword(ad, above_post, below_post) {
            v.keyword_adj += 1;
        }
    }

    v
}

fn check_invariants(items: &[FeedItem], blender_name: &str, scenario_idx: usize) -> bool {
    if items.is_empty() {
        return true;
    }

    if matches!(items.first(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        eprintln!("[{blender_name}] scenario {scenario_idx}: ad at position 0!");
        return false;
    }
    if matches!(items.last(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        eprintln!("[{blender_name}] scenario {scenario_idx}: ad at last position!");
        return false;
    }

    let positions = ad_positions(items);
    for w in positions.windows(2) {
        let between = (w[1] - w[0] - 1) as usize;
        if between < 2 {
            eprintln!(
                "[{blender_name}] scenario {scenario_idx}: only {between} items between ads at {} and {}",
                w[0], w[1]
            );
            return false;
        }
    }

    true
}

struct BlenderStats {
    name: &'static str,
    total_ads_placed: usize,
    total_ads_available: usize,
    total_posts: usize,
    total_items: usize,
    invariant_violations: usize,
    total_spacing_sum: f64,
    total_spacing_count: usize,
    ads_with_medrisk_adj: usize,
    bsr_low_with_lowrisk_adj: usize,
    bsr_low_available: usize,
    bsr_low_placed: usize,
    handle_adj: usize,
    keyword_adj: usize,
}

impl BlenderStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            total_ads_placed: 0,
            total_ads_available: 0,
            total_posts: 0,
            total_items: 0,
            invariant_violations: 0,
            total_spacing_sum: 0.0,
            total_spacing_count: 0,
            ads_with_medrisk_adj: 0,
            bsr_low_with_lowrisk_adj: 0,
            bsr_low_available: 0,
            bsr_low_placed: 0,
            handle_adj: 0,
            keyword_adj: 0,
        }
    }

    fn ad_fill_rate(&self) -> f64 {
        if self.total_ads_available == 0 {
            return 0.0;
        }
        self.total_ads_placed as f64 / self.total_ads_available as f64 * 100.0
    }

    fn ad_load_pct(&self) -> f64 {
        if self.total_items == 0 {
            return 0.0;
        }
        self.total_ads_placed as f64 / self.total_items as f64 * 100.0
    }

    fn avg_spacing(&self) -> f64 {
        if self.total_spacing_count == 0 {
            return 0.0;
        }
        self.total_spacing_sum / self.total_spacing_count as f64
    }

    fn medrisk_adj_pct(&self) -> f64 {
        if self.total_ads_placed == 0 {
            return 0.0;
        }
        self.ads_with_medrisk_adj as f64 / self.total_ads_placed as f64 * 100.0
    }

    fn bsr_low_lowrisk_adj_pct(&self) -> f64 {
        if self.bsr_low_placed == 0 {
            return 0.0;
        }
        self.bsr_low_with_lowrisk_adj as f64 / self.bsr_low_placed as f64 * 100.0
    }

    fn bsr_low_drop_pct(&self) -> f64 {
        if self.bsr_low_available == 0 {
            return 0.0;
        }
        (1.0 - self.bsr_low_placed as f64 / self.bsr_low_available as f64) * 100.0
    }
}

type BlendFn = fn(Vec<ScoredPost>, Vec<AdIndexInfo>, usize) -> Vec<FeedItem>;

fn safe_gap_blend(
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
) -> Vec<FeedItem> {
    safe_gap_blender::blend_impl(posts, ads, min_posts, RESULT_SIZE)
}

#[test]
fn blender_comparison_report() {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let scenarios = generate_scenarios(&mut rng);

    let blenders: Vec<(&str, BlendFn)> = vec![
        ("safe_gap", safe_gap_blend),
        ("partition_organic", partition_organic_blender::blend_impl),
    ];

    let mut all_stats: Vec<BlenderStats> = blenders
        .iter()
        .map(|(name, _)| BlenderStats::new(name))
        .collect();

    for (scenario_idx, scenario) in scenarios.iter().enumerate() {
        for (blender_idx, (_name, blend_fn)) in blenders.iter().enumerate() {
            let posts = scenario.posts.clone();
            let ads = scenario.ads.clone();
            let num_ads_available = ads.len();

            let result = blend_fn(posts, ads, 0);

            let ads_placed = ad_count(&result);
            let posts_in_result = post_count(&result);

            if !check_invariants(&result, all_stats[blender_idx].name, scenario_idx) {
                all_stats[blender_idx].invariant_violations += 1;
            }

            let positions = ad_positions(&result);
            for w in positions.windows(2) {
                let spacing = (w[1] - w[0] - 1) as f64;
                all_stats[blender_idx].total_spacing_sum += spacing;
                all_stats[blender_idx].total_spacing_count += 1;
            }

            let v = count_adjacency_violations(&result, &scenario.ads);
            all_stats[blender_idx].ads_with_medrisk_adj += v.med_risk_adj;
            all_stats[blender_idx].bsr_low_with_lowrisk_adj += v.bsr_low_low_risk_adj;
            all_stats[blender_idx].bsr_low_available += v.bsr_low_available;
            all_stats[blender_idx].bsr_low_placed += v.bsr_low_placed;
            all_stats[blender_idx].handle_adj += v.handle_adj;
            all_stats[blender_idx].keyword_adj += v.keyword_adj;

            all_stats[blender_idx].total_ads_placed += ads_placed;
            all_stats[blender_idx].total_ads_available += num_ads_available;
            all_stats[blender_idx].total_posts += posts_in_result;
            all_stats[blender_idx].total_items += result.len();
        }
    }

    eprintln!();
    eprintln!(
        "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║                                              ADS BLENDER COMPARISON REPORT                                                         ║"
    );
    eprintln!(
        "║  {NUM_SCENARIOS} scenarios — posts: ~{:.0}% MedRisk, ~{:.0}% LowRisk, ~{:.0}% Safe — ads: ~{:.0}% BSR_LOW, ~{:.0}% BSR_IAS, ~{:.0}% BSR_NORMAL          ║",
        UNSAFE_POST_RATE * 100.0,
        LOW_RISK_POST_RATE * 100.0,
        (1.0 - UNSAFE_POST_RATE - LOW_RISK_POST_RATE) * 100.0,
        BSR_LOW_AD_RATE * 100.0,
        BSR_IAS_AD_RATE * 100.0,
        (1.0 - BSR_LOW_AD_RATE - BSR_IAS_AD_RATE - BSR_NO_RISK_AD_RATE) * 100.0,
    );
    eprintln!(
        "╠══════════════════════╦═════════╦═════════╦═════════╦═════════╦═════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╣"
    );
    eprintln!(
        "║ Blender              ║ AdPlace ║ AdAvail ║  Fill%  ║  Load%  ║ AvgSpc  ║ MedAdj%  ║ LR→BSR%  ║ BSRDrop% ║ BSRPlcd  ║ HndlAdj  ║  KwAdj   ║"
    );
    eprintln!(
        "╠══════════════════════╬═════════╬═════════╬═════════╬═════════╬═════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╣"
    );
    for stats in &all_stats {
        eprintln!(
            "║ {:<20} ║ {:>7} ║ {:>7} ║ {:>6.1}% ║ {:>6.1}% ║ {:>7.2} ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║ {:>4}/{:<4}║ {:>8} ║ {:>8} ║",
            stats.name,
            stats.total_ads_placed,
            stats.total_ads_available,
            stats.ad_fill_rate(),
            stats.ad_load_pct(),
            stats.avg_spacing(),
            stats.medrisk_adj_pct(),
            stats.bsr_low_lowrisk_adj_pct(),
            stats.bsr_low_drop_pct(),
            stats.bsr_low_placed,
            stats.bsr_low_available,
            stats.handle_adj,
            stats.keyword_adj,
        );
    }
    eprintln!(
        "╚══════════════════════╩═════════╩═════════╩═════════╩═════════╩═════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╝"
    );
    eprintln!();
    eprintln!("Legend: Fill% = ads placed / ads available");
    eprintln!("        Load% = ads placed / total items (ad load in timeline)");
    eprintln!("        AvgSpc = average number of posts between consecutive ads");
    eprintln!("        MedAdj% = % of placed ads adjacent to MediumRisk post");
    eprintln!(
        "        LR→BSR% = % of placed BSR_LOW/IAS ads adjacent to LowRisk post (should be 0 for partition_organic)"
    );
    eprintln!("        BSRDrop% = % of BSR_LOW/IAS ads dropped (not placed)");
    eprintln!("        BSRPlcd = BSR_LOW/IAS ads placed / available");
    eprintln!(
        "        HndlAdj = placed ads adjacent to an excluded handle (should be 0 for partition_organic)"
    );
    eprintln!(
        "        KwAdj = placed ads adjacent to an excluded keyword match (should be 0 for partition_organic)"
    );
    eprintln!();

    for stats in &all_stats {
        assert_eq!(
            stats.invariant_violations, 0,
            "{} had {} invariant violations",
            stats.name, stats.invariant_violations
        );
    }

    let po_stats = all_stats
        .iter()
        .find(|s| s.name == "partition_organic")
        .expect("partition_organic blender not found in stats");
    assert_eq!(
        po_stats.bsr_low_with_lowrisk_adj, 0,
        "partition_organic had {} BSR_LOW/IAS ads adjacent to LowRisk posts!",
        po_stats.bsr_low_with_lowrisk_adj
    );
    assert_eq!(
        po_stats.handle_adj, 0,
        "partition_organic had {} ads adjacent to excluded handles!",
        po_stats.handle_adj
    );
    assert_eq!(
        po_stats.keyword_adj, 0,
        "partition_organic had {} ads adjacent to excluded keyword matches!",
        po_stats.keyword_adj
    );
}
