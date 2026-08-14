use std::collections::HashMap;

pub type PostId = i64;
pub type AuthorId = i64;

#[derive(Debug, Clone)]
pub enum IndexRecord {
    Core {
        post_id: PostId,
        author_id: AuthorId,
        index_name: String,
    },

    Topic {
        post_id: PostId,
        author_id: AuthorId,
        index_name: String,
        topic_entity_ids: Vec<i64>,
    },

    Metadata {
        post_id: PostId,
        author_id: AuthorId,
        has_video: bool,
        has_image: bool,
        video_duration_ms: i64,
        author_followers_count: i64,
        engagement: EngagementCounts,
    },

    MmMetadata {
        post_id: PostId,
        author_id: AuthorId,
        has_video: bool,
        has_image: bool,
        video_duration_ms: i64,
        author_followers_count: i64,
        engagement: EngagementCounts,
        mm_emb_v1: Option<Vec<f32>>,
        mm_emb_v3: Option<Vec<f32>>,
    },

    Ads {
        post_id: PostId,
        author_id: AuthorId,
    },

    Sid {
        post_id: PostId,
        author_id: AuthorId,
        index_name: String,
        post_sid: Vec<i32>,
    },

    Analysis {
        post_id: PostId,
        author_id: AuthorId,
        viewer_id: i64,
        served_type: Option<String>,
        scores: HashMap<String, f64>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct EngagementCounts {
    pub retweet_count: i64,
    pub reply_count: i64,
    pub fav_count: i64,
    pub quote_count: i64,
    pub bookmark_count: i64,
    pub view_count: i64,
    pub not_interested_in_count: i64,
    pub report_count: i64,
    pub block_count: i64,
}

impl IndexRecord {
    pub fn post_id(&self) -> PostId {
        match self {
            Self::Core { post_id, .. }
            | Self::Topic { post_id, .. }
            | Self::Metadata { post_id, .. }
            | Self::MmMetadata { post_id, .. }
            | Self::Ads { post_id, .. }
            | Self::Sid { post_id, .. }
            | Self::Analysis { post_id, .. } => *post_id,
        }
    }

    pub fn author_id(&self) -> AuthorId {
        match self {
            Self::Core { author_id, .. }
            | Self::Topic { author_id, .. }
            | Self::Metadata { author_id, .. }
            | Self::MmMetadata { author_id, .. }
            | Self::Ads { author_id, .. }
            | Self::Sid { author_id, .. }
            | Self::Analysis { author_id, .. } => *author_id,
        }
    }

    pub fn index_name(&self) -> &str {
        match self {
            Self::Core { index_name, .. }
            | Self::Topic { index_name, .. }
            | Self::Sid { index_name, .. } => index_name,
            Self::Metadata { .. } => "metadata",
            Self::MmMetadata { .. } => "mm_emb_metadata",
            Self::Ads { .. } => "ads",
            Self::Analysis { .. } => "analysis",
        }
    }
}
