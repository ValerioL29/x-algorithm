pub const ANY_CANDIDATE: usize = 0;
pub const RETWEET: usize = 1;
pub const REPLY: usize = 2;
pub const VIDEO: usize = 5;
pub const SUBSCRIPTION_POST: usize = 10;
pub const NEAR_EMPTY: usize = 23;
pub const EMPTY_REQUEST: usize = 32;
pub const HAS_ANCESTORS: usize = 37;
pub const FULL_SCORING_SUCCEEDED: usize = 38;
pub const SERVED_SIZE_LESS_THAN_20: usize = 39;
pub const SERVED_SIZE_LESS_THAN_10: usize = 40;
pub const SERVED_SIZE_LESS_THAN_5: usize = 41;
pub const IN_NETWORK: usize = 80;
pub const VIDEO_LTE_10_SEC: usize = 151;
pub const VIDEO_BT_10_60_SEC: usize = 152;
pub const VIDEO_GT_60_SEC: usize = 153;
pub const TWEET_AGE_LTE_30_MINUTES: usize = 154;
pub const TWEET_AGE_LTE_1_HOUR: usize = 155;
pub const TWEET_AGE_LTE_6_HOURS: usize = 156;
pub const TWEET_AGE_LTE_12_HOURS: usize = 157;
pub const TWEET_AGE_GTE_24_HOURS: usize = 158;
pub const AUTHOR_FOLLOWERS_0_100: usize = 309;
pub const AUTHOR_FOLLOWERS_100_1K: usize = 310;
pub const AUTHOR_FOLLOWERS_1K_10K: usize = 311;
pub const AUTHOR_FOLLOWERS_10K_100K: usize = 312;
pub const AUTHOR_FOLLOWERS_100K_1M: usize = 313;
pub const AUTHOR_FOLLOWERS_1M_PLUS: usize = 314;

pub const TWEET_TYPE_PREDICATES: &[(usize, &str)] = &[
    (ANY_CANDIDATE, "with_candidate"),
    (RETWEET, "retweet"),
    (REPLY, "reply"),
    (VIDEO, "video"),
    (SUBSCRIPTION_POST, "has_exclusive_conversation_author_id"),
    (NEAR_EMPTY, "near_empty"),
    (EMPTY_REQUEST, "empty_request"),
    (HAS_ANCESTORS, "has_ancestors"),
    (FULL_SCORING_SUCCEEDED, "full_scoring_succeeded"),
    (SERVED_SIZE_LESS_THAN_20, "served_size_less_than_20"),
    (SERVED_SIZE_LESS_THAN_10, "served_size_less_than_10"),
    (SERVED_SIZE_LESS_THAN_5, "served_size_less_than_5"),
    (IN_NETWORK, "in_network"),
    (VIDEO_LTE_10_SEC, "video_lte_10_sec"),
    (VIDEO_BT_10_60_SEC, "video_bt_10_60_sec"),
    (VIDEO_GT_60_SEC, "video_gt_60_sec"),
    (TWEET_AGE_LTE_30_MINUTES, "tweet_age_lte_30_minutes"),
    (TWEET_AGE_LTE_1_HOUR, "tweet_age_lte_1_hour"),
    (TWEET_AGE_LTE_6_HOURS, "tweet_age_lte_6_hours"),
    (TWEET_AGE_LTE_12_HOURS, "tweet_age_lte_12_hours"),
    (TWEET_AGE_GTE_24_HOURS, "tweet_age_gte_24_hours"),
    (AUTHOR_FOLLOWERS_0_100, "author_followers_0_100"),
    (AUTHOR_FOLLOWERS_100_1K, "author_followers_100_1k"),
    (AUTHOR_FOLLOWERS_1K_10K, "author_followers_1k_10k"),
    (AUTHOR_FOLLOWERS_10K_100K, "author_followers_10k_100k"),
    (AUTHOR_FOLLOWERS_100K_1M, "author_followers_100k_1m"),
    (AUTHOR_FOLLOWERS_1M_PLUS, "author_followers_1m_plus"),
];

pub fn bitset_get(bytes: &[u8], bit_index: usize) -> bool {
    let byte_index = bit_index / 8;
    let bit_offset = bit_index % 8;
    bytes
        .get(byte_index)
        .is_some_and(|b| b & (1 << bit_offset) != 0)
}
