pub use xai_x_thrift::tweet_safety_label::{SafetyLabel, SafetyLabelType};

use std::collections::{HashMap, HashSet};
use xai_visibility_filtering_proto as vf_pb;

#[derive(Clone, Debug, Default)]
pub struct SafetyLabelMap {
    pub labels: HashMap<SafetyLabelType, SafetyLabel>,
    label_types: HashSet<SafetyLabelType>,
}

impl SafetyLabelMap {
    pub fn new(labels: HashMap<SafetyLabelType, SafetyLabel>) -> Self {
        let label_types = labels.keys().copied().collect();
        Self {
            labels,
            label_types,
        }
    }

    pub fn from_proto_label_types(proto: &vf_pb::SafetyLabelMap) -> Self {
        let labels = proto
            .labels
            .keys()
            .map(|label_type| (SafetyLabelType(*label_type), SafetyLabel::default()))
            .collect();
        Self::new(labels)
    }

    #[inline]
    pub fn has_label(&self, label_type: SafetyLabelType) -> bool {
        self.label_types.contains(&label_type)
    }
}
