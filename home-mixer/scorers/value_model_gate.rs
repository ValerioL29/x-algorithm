use crate::models::query::ScoredPostsQuery;
use crate::params::{
    DwellRegretGateBias, DwellRegretGateHysteresisBand, DwellRegretGateThreshold,
    DwellRegretGateWeights,
};
use arrow::array::{Array, BooleanArray, FixedSizeListArray, Int64Array};
use arrow::ipc::reader::StreamReader;
use xai_recsys_proto::ActionName;
use xai_stats_receiver::{global_stats_receiver, HistogramBuckets};

const METRIC_PREFIX: &str = "ValueModelGate";

const N_FEATURES: usize = 19;

const FEATURE_NAMES: [&str; N_FEATURES] = [
    "seq_len",
    "n_fav",
    "n_reply",
    "n_rt_quote",
    "n_vqv",
    "n_click",
    "n_bm_share",
    "n_profile_follow",
    "n_photo",
    "n_negfb",
    "n_7d",
    "n_1d",
    "active_days",
    "active_days_7d",
    "days_since_last",
    "span_days",
    "followers",
    "followings",
    "account_age_years",
];

const LOG1P_FEATURES: [bool; N_FEATURES] = [
    true, true, true, true, true, true, true, true, true, true, true, true, false, false, false,
    false, true, true, false,
];

const DAY_MS: i64 = 86_400_000;
const WINDOW_DAYS: f64 = 28.0;
const WINDOW_MS: i64 = 28 * DAY_MS;
const FOLLOWERS_CAP: f64 = 1e7;
const FOLLOWINGS_CAP: f64 = 1e6;
const ACCOUNT_AGE_CAP_YEARS: f64 = 25.0;
const PRE_SNOWFLAKE_AGE_YEARS: f64 = 16.0;
const COL_IMPRESSED_TIME_MS: &str = "impressedTimeMs";
const COL_ACTION_NAME_MULTI_HOT: &str = "actionNameMultiHot";

const FAV_ACTIONS: [ActionName; 1] = [ActionName::ServerTweetFav];
const REPLY_ACTIONS: [ActionName; 1] = [ActionName::ServerTweetReply];
const RT_QUOTE_ACTIONS: [ActionName; 2] =
    [ActionName::ServerTweetRetweet, ActionName::ServerTweetQuote];
const VQV_ACTIONS: [ActionName; 2] = [
    ActionName::ClientTweetVideoQualityView,
    ActionName::ClientQuotedTweetVideoQualityView,
];
const CLICK_ACTIONS: [ActionName; 2] = [
    ActionName::ClientTweetClick,
    ActionName::ClientQuotedTweetClick,
];
const BM_SHARE_ACTIONS: [ActionName; 4] = [
    ActionName::ClientTweetBookmark,
    ActionName::ClientTweetShare,
    ActionName::ClientTweetShareViaCopyLink,
    ActionName::ClientTweetClickSendViaDirectMessage,
];
const PROFILE_FOLLOW_ACTIONS: [ActionName; 2] = [
    ActionName::ClientTweetClickProfile,
    ActionName::ClientTweetFollowAuthor,
];
const PHOTO_ACTIONS: [ActionName; 2] = [
    ActionName::ClientTweetPhotoExpand,
    ActionName::ClientQuotedTweetPhotoExpand,
];
const NEGFB_ACTIONS: [ActionName; 6] = [
    ActionName::ClientTweetNotInterestedIn,
    ActionName::ClientTweetSeeFewer,
    ActionName::ClientTweetNotRelevant,
    ActionName::ClientTweetReport,
    ActionName::ClientTweetBlockAuthor,
    ActionName::ClientTweetMuteAuthor,
];

#[derive(Debug, Default, PartialEq)]
pub(crate) struct GateFeatures {
    pub values: [f64; N_FEATURES],
}

pub(crate) struct GateModel {
    weights: [f64; N_FEATURES],
    bias: f64,
    threshold: f64,
    hysteresis_band: f64,
}

impl GateModel {
    pub(crate) fn from_params(params: &xai_feature_switches::Params) -> Option<Self> {
        let spec: String = params.get(DwellRegretGateWeights);
        let mut weights = [f64::NAN; N_FEATURES];
        for entry in spec.split(',') {
            let (name, value) = entry.split_once(':')?;
            let idx = FEATURE_NAMES.iter().position(|f| *f == name.trim())?;
            weights[idx] = value.trim().parse().ok()?;
        }
        if weights.iter().any(|w| w.is_nan()) {
            return None;
        }
        Some(Self {
            weights,
            bias: params.get(DwellRegretGateBias),
            threshold: params.get(DwellRegretGateThreshold),
            hysteresis_band: params.get(DwellRegretGateHysteresisBand).max(0.0),
        })
    }

    pub(crate) fn score(&self, features: &GateFeatures) -> f64 {
        let mut score = self.bias;
        for (i, &v) in features.values.iter().enumerate() {
            let x = if LOG1P_FEATURES[i] {
                v.max(0.0).ln_1p()
            } else {
                v
            };
            score += self.weights[i] * x;
        }
        score
    }

    pub(crate) fn serve_new_scoring(&self, query: &ScoredPostsQuery) -> bool {
        let features = extract_features(query);
        let score = self.score(&features);
        let margin = score - self.threshold;
        let decision = if margin.abs() <= self.hysteresis_band {
            query.user_id.wrapping_mul(0x9E3779B97F4A7C15) & 1 == 0
        } else {
            margin > 0.0
        };
        record_stats(query, score, decision);
        decision
    }
}

fn record_stats(query: &ScoredPostsQuery, score: f64, serve_new: bool) {
    let Some(receiver) = global_stats_receiver() else {
        return;
    };
    let has_sequence = query.columnar_scoring_sequence.is_some();
    receiver.incr(
        &format!("{METRIC_PREFIX}.decision"),
        &[
            ("serve_new", if serve_new { "true" } else { "false" }),
            ("has_sequence", if has_sequence { "true" } else { "false" }),
        ],
        1,
    );
    receiver.observe(
        &format!("{METRIC_PREFIX}.score"),
        &[],
        (score.clamp(-5.0, 5.0) + 5.0) / 10.0,
        HistogramBuckets::Bucket0To1,
    );
}

pub(crate) fn extract_features(query: &ScoredPostsQuery) -> GateFeatures {
    let now_ms = query.request_time_ms;
    let mut f = GateFeatures::default();
    f.values[14] = WINDOW_DAYS;

    if let Some(bytes) = &query.columnar_scoring_sequence {
        if let Some(rows) = parse_sequence(bytes) {
            accumulate_sequence_features(&mut f, &rows, now_ms);
        } else if let Some(receiver) = global_stats_receiver() {
            receiver.incr(&format!("{METRIC_PREFIX}.parse_error"), &[], 1);
        }
    }

    f.values[16] =
        (query.user_features.follower_count.unwrap_or(0).max(0) as f64).min(FOLLOWERS_CAP);
    f.values[17] = (query.user_features.followed_user_ids.len() as f64).min(FOLLOWINGS_CAP);
    f.values[18] = account_age_years(query.user_id, now_ms);
    f
}

struct SequenceRow {
    ts_ms: i64,
    action_mask: u64,
}

fn parse_sequence(bytes: &bytes::Bytes) -> Option<Vec<SequenceRow>> {
    let reader = StreamReader::try_new(std::io::Cursor::new(bytes.as_ref()), None).ok()?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch.ok()?;
        let ts = batch
            .column_by_name(COL_IMPRESSED_TIME_MS)?
            .as_any()
            .downcast_ref::<Int64Array>()?;
        let multi_hot = batch
            .column_by_name(COL_ACTION_NAME_MULTI_HOT)?
            .as_any()
            .downcast_ref::<FixedSizeListArray>()?;
        let hot_values = multi_hot.values().as_any().downcast_ref::<BooleanArray>()?;
        let vocab = multi_hot.value_length() as usize;
        for i in 0..batch.num_rows() {
            let mut mask = 0u64;
            let base = i * vocab;
            for (j, _) in ALL_TRACKED_ACTIONS.iter().enumerate() {
                let idx = ALL_TRACKED_INDICES[j];
                if idx < vocab && hot_values.value(base + idx) {
                    mask |= 1 << j;
                }
            }
            rows.push(SequenceRow {
                ts_ms: ts.value(i),
                action_mask: mask,
            });
        }
    }
    Some(rows)
}

const N_TRACKED: usize = 22;

const ALL_TRACKED_ACTIONS: [ActionName; N_TRACKED] = [
    ActionName::ServerTweetFav,
    ActionName::ServerTweetReply,
    ActionName::ServerTweetRetweet,
    ActionName::ServerTweetQuote,
    ActionName::ClientTweetVideoQualityView,
    ActionName::ClientQuotedTweetVideoQualityView,
    ActionName::ClientTweetClick,
    ActionName::ClientQuotedTweetClick,
    ActionName::ClientTweetBookmark,
    ActionName::ClientTweetShare,
    ActionName::ClientTweetShareViaCopyLink,
    ActionName::ClientTweetClickSendViaDirectMessage,
    ActionName::ClientTweetClickProfile,
    ActionName::ClientTweetFollowAuthor,
    ActionName::ClientTweetPhotoExpand,
    ActionName::ClientQuotedTweetPhotoExpand,
    ActionName::ClientTweetNotInterestedIn,
    ActionName::ClientTweetSeeFewer,
    ActionName::ClientTweetNotRelevant,
    ActionName::ClientTweetReport,
    ActionName::ClientTweetBlockAuthor,
    ActionName::ClientTweetMuteAuthor,
];

const ALL_TRACKED_INDICES: [usize; N_TRACKED] = tracked_indices();

const fn tracked_indices() -> [usize; N_TRACKED] {
    let mut out = [0usize; N_TRACKED];
    let mut i = 0;
    while i < N_TRACKED {
        out[i] = ALL_TRACKED_ACTIONS[i] as usize;
        i += 1;
    }
    out
}

fn group_mask(actions: &[ActionName]) -> u64 {
    let mut mask = 0u64;
    for a in actions {
        if let Some(j) = ALL_TRACKED_ACTIONS.iter().position(|t| t == a) {
            mask |= 1 << j;
        }
    }
    mask
}

fn accumulate_sequence_features(f: &mut GateFeatures, rows: &[SequenceRow], now_ms: i64) {
    let fav = group_mask(&FAV_ACTIONS);
    let reply = group_mask(&REPLY_ACTIONS);
    let rt_quote = group_mask(&RT_QUOTE_ACTIONS);
    let vqv = group_mask(&VQV_ACTIONS);
    let click = group_mask(&CLICK_ACTIONS);
    let bm_share = group_mask(&BM_SHARE_ACTIONS);
    let profile_follow = group_mask(&PROFILE_FOLLOW_ACTIONS);
    let photo = group_mask(&PHOTO_ACTIONS);
    let negfb = group_mask(&NEGFB_ACTIONS);

    let mut first_ts = i64::MAX;
    let mut last_ts = i64::MIN;
    let mut days = std::collections::HashSet::new();
    let mut days_7d = std::collections::HashSet::new();

    for row in rows {
        if row.ts_ms < now_ms - WINDOW_MS || row.ts_ms > now_ms {
            continue;
        }
        f.values[0] += 1.0;
        for (idx, mask) in [
            (1, fav),
            (2, reply),
            (3, rt_quote),
            (4, vqv),
            (5, click),
            (6, bm_share),
            (7, profile_follow),
            (8, photo),
            (9, negfb),
        ] {
            if row.action_mask & mask != 0 {
                f.values[idx] += 1.0;
            }
        }
        if row.ts_ms >= now_ms - 7 * DAY_MS {
            f.values[10] += 1.0;
            days_7d.insert(row.ts_ms.div_euclid(DAY_MS));
        }
        if row.ts_ms >= now_ms - DAY_MS {
            f.values[11] += 1.0;
        }
        days.insert(row.ts_ms.div_euclid(DAY_MS));
        first_ts = first_ts.min(row.ts_ms);
        last_ts = last_ts.max(row.ts_ms);
    }

    f.values[12] = days.len() as f64;
    f.values[13] = days_7d.len() as f64;
    if last_ts == i64::MIN {
        f.values[14] = WINDOW_DAYS;
        f.values[15] = 0.0;
    } else {
        f.values[14] = (((now_ms - last_ts) as f64) / DAY_MS as f64).min(WINDOW_DAYS);
        f.values[15] = ((now_ms - first_ts) as f64) / DAY_MS as f64;
    }
}

fn account_age_years(user_id: u64, now_ms: i64) -> f64 {
    match xai_candidate_pipeline::component_library::utils::time_from_id_opt(user_id) {
        Some(created) => {
            let created_ms = created
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as i64)
                .unwrap_or(now_ms);
            (((now_ms - created_ms) as f64) / (365.25 * DAY_MS as f64))
                .clamp(0.0, ACCOUNT_AGE_CAP_YEARS)
        }
        None => PRE_SNOWFLAKE_AGE_YEARS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    const VOCAB: usize = 64;
    const NOW_MS: i64 = 1_785_000_000_000;

    const FULL_WEIGHT_SPEC: &str = "seq_len:0.53,n_fav:-0.08,n_reply:0.49,n_rt_quote:0.06,n_vqv:-0.07,n_click:-0.18,n_bm_share:-0.17,n_profile_follow:-0.22,n_photo:-0.11,n_negfb:0.03,n_7d:-0.08,n_1d:-0.24,active_days:0.05,active_days_7d:-0.13,days_since_last:-0.03,span_days:-0.05,followers:-0.07,followings:0.06,account_age_years:-0.05";

    const GATE_WEIGHTS_ALL_ZERO: &str = "seq_len:0,n_fav:0,n_reply:0,n_rt_quote:0,n_vqv:0,n_click:0,n_bm_share:0,n_profile_follow:0,n_photo:0,n_negfb:0,n_7d:0,n_1d:0,active_days:0,active_days_7d:0,days_since_last:0,span_days:0,followers:0,followings:0,account_age_years:0";

    fn query_with_flags(flags: &[(&str, &str)]) -> ScoredPostsQuery {
        let mut query = ScoredPostsQuery::default();
        let fs = xai_feature_switches::FeatureSwitches::new(vec![]).unwrap();
        let mut results =
            fs.match_recipient(&xai_feature_switches::RecipientBuilder::new().build());
        for (key, value) in flags {
            results.override_fs(key.to_string(), value);
        }
        query.params = results.into();
        query.request_time_ms = NOW_MS;
        query
    }

    fn sequence_bytes(rows: &[(i64, &[ActionName])]) -> bytes::Bytes {
        let list_field = Arc::new(Field::new("item", DataType::Boolean, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new(COL_IMPRESSED_TIME_MS, DataType::Int64, false),
            Field::new(
                COL_ACTION_NAME_MULTI_HOT,
                DataType::FixedSizeList(list_field.clone(), VOCAB as i32),
                false,
            ),
        ]));
        let ts = Int64Array::from(rows.iter().map(|(t, _)| *t).collect::<Vec<_>>());
        let mut hot = vec![false; rows.len() * VOCAB];
        for (i, (_, actions)) in rows.iter().enumerate() {
            for a in *actions {
                hot[i * VOCAB + *a as usize] = true;
            }
        }
        let values = BooleanArray::from(hot);
        let multi_hot = FixedSizeListArray::new(list_field, VOCAB as i32, Arc::new(values), None);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(ts) as ArrayRef, Arc::new(multi_hot) as ArrayRef],
        )
        .unwrap();
        let mut buf = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut buf, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }
        bytes::Bytes::from(buf)
    }

    #[test]
    fn extract_features_counts_groups_and_windows() {
        let mut query = query_with_flags(&[]);
        query.user_features.follower_count = Some(500);
        query.user_features.followed_user_ids = vec![1, 2, 3];
        query.columnar_scoring_sequence = Some(sequence_bytes(&[
            (NOW_MS - 3_600_000, &[ActionName::ServerTweetFav]),
            (
                NOW_MS - 10 * DAY_MS,
                &[
                    ActionName::ClientTweetVideoQualityView,
                    ActionName::ClientTweetClick,
                ],
            ),
            (NOW_MS - 40 * DAY_MS, &[ActionName::ServerTweetFav]),
        ]));

        let f = extract_features(&query);
        assert_eq!(f.values[0], 2.0);
        assert_eq!(f.values[1], 1.0);
        assert_eq!(f.values[4], 1.0);
        assert_eq!(f.values[5], 1.0);
        assert_eq!(f.values[10], 1.0);
        assert_eq!(f.values[11], 1.0);
        assert_eq!(f.values[12], 2.0);
        assert_eq!(f.values[13], 1.0);
        assert!((f.values[14] - 1.0 / 24.0).abs() < 1e-9);
        assert!((f.values[15] - 10.0).abs() < 1e-9);
        assert_eq!(f.values[16], 500.0);
        assert_eq!(f.values[17], 3.0);
    }

    #[test]
    fn extract_features_without_sequence_uses_missing_defaults() {
        let query = query_with_flags(&[]);
        let f = extract_features(&query);
        assert_eq!(f.values[0], 0.0);
        assert_eq!(f.values[14], WINDOW_DAYS);
        assert_eq!(f.values[15], 0.0);
    }

    #[test]
    fn gate_model_parses_full_weight_spec_and_scores() {
        let query = query_with_flags(&[
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                FULL_WEIGHT_SPEC,
            ),
            ("rust_home_mixer_dwell_regret_gate_bias", "0.25"),
        ]);
        let gate = GateModel::from_params(&query.params).expect("full weight spec parses");
        let f = extract_features(&query);
        let score = gate.score(&f);
        assert!(score.is_finite());
    }

    #[test]
    fn gate_model_rejects_malformed_or_incomplete_weights() {
        let query = query_with_flags(&[(
            "rust_home_mixer_dwell_regret_gate_weights",
            "seq_len:0.5,bogus",
        )]);
        assert!(GateModel::from_params(&query.params).is_none());
        let query =
            query_with_flags(&[("rust_home_mixer_dwell_regret_gate_weights", "seq_len:0.5")]);
        assert!(GateModel::from_params(&query.params).is_none());
    }

    #[test]
    fn linear_score_applies_log_transform_and_bias() {
        let query = query_with_flags(&[
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                "seq_len:1.0,n_fav:0,n_reply:0,n_rt_quote:0,n_vqv:0,n_click:0,n_bm_share:0,n_profile_follow:0,n_photo:0,n_negfb:0,n_7d:0,n_1d:0,active_days:0,active_days_7d:0,days_since_last:2.0,span_days:0,followers:0,followings:0,account_age_years:0",
            ),
            ("rust_home_mixer_dwell_regret_gate_bias", "0.25"),
        ]);
        let gate = GateModel::from_params(&query.params).unwrap();
        let mut f = GateFeatures::default();
        f.values[0] = (std::f64::consts::E) - 1.0;
        f.values[14] = 3.0;
        assert!((gate.score(&f) - (1.0 + 6.0 + 0.25)).abs() < 1e-9);
    }

    #[test]
    fn threshold_and_hysteresis_decide_deterministically() {
        let mut query = query_with_flags(&[
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                GATE_WEIGHTS_ALL_ZERO,
            ),
            ("rust_home_mixer_dwell_regret_gate_bias", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_hysteresis_band", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_threshold", "1000.0"),
        ]);
        query.user_id = 12345;
        let gate = GateModel::from_params(&query.params).unwrap();
        assert!(!gate.serve_new_scoring(&query));

        let mut query = query_with_flags(&[
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                GATE_WEIGHTS_ALL_ZERO,
            ),
            ("rust_home_mixer_dwell_regret_gate_bias", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_hysteresis_band", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_threshold", "-1000.0"),
        ]);
        query.user_id = 12345;
        let gate = GateModel::from_params(&query.params).unwrap();
        assert!(gate.serve_new_scoring(&query));

        let mut query = query_with_flags(&[
            (
                "rust_home_mixer_dwell_regret_gate_weights",
                GATE_WEIGHTS_ALL_ZERO,
            ),
            ("rust_home_mixer_dwell_regret_gate_bias", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_threshold", "0.0"),
            ("rust_home_mixer_dwell_regret_gate_hysteresis_band", "1e9"),
        ]);
        query.user_id = 4;
        let gate = GateModel::from_params(&query.params).unwrap();
        let first = gate.serve_new_scoring(&query);
        for _ in 0..5 {
            assert_eq!(gate.serve_new_scoring(&query), first);
        }
    }
}
