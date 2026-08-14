use std::collections::HashSet;
use xai_x_thrift::user_labels::LabelValue;

#[derive(Clone, Debug, Default)]
pub struct AuthorFeatures {
    pub is_suspended: bool,
    pub is_deactivated: bool,
    pub is_protected: bool,
    pub is_nsfw_user: bool,
    pub is_nsfw_admin: bool,
    pub is_erased: bool,
    pub is_offboarded: bool,
    pub user_labels: UserLabelSet,
}

#[derive(Clone, Debug, Default)]
pub struct UserLabelSet {
    labels: HashSet<LabelValue>,
}

impl UserLabelSet {
    pub fn new(labels: HashSet<LabelValue>) -> Self {
        Self { labels }
    }

    #[inline]
    pub fn has_label(&self, label: LabelValue) -> bool {
        self.labels.contains(&label)
    }
}
