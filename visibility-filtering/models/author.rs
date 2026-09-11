#[derive(Clone, Copy, Debug, Default)]
pub struct AuthorFeatures {
    pub is_suspended: bool,
    pub is_deactivated: bool,
    pub is_protected: bool,
    pub is_nsfw_user: bool,
    pub is_nsfw_admin: bool,
    pub is_erased: bool,
    pub is_offboarded: bool,
    pub user_labels: AuthorLabelSet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AuthorLabel {
    NsfwHighRecall,
    NsfwHighPrecision,
    NsfwNearPerfect,
    NsfwAvatarImage,
    NsfwBannerImage,
    SpamHighRecall,
    AbusiveHighRecall,
    Compromised,
    ReadOnly,
    ImpersonationHighPrecision,
    DoNotAmplify,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AuthorLabelSet(u64);

impl AuthorLabelSet {
    #[inline]
    pub fn insert(&mut self, label: AuthorLabel) {
        self.0 |= 1 << label as u8;
    }

    #[inline]
    pub fn has_label(self, label: AuthorLabel) -> bool {
        self.0 & (1 << label as u8) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn author_features_is_copy_and_16_bytes_with_option_niche() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<AuthorFeatures>();
        assert_eq!(std::mem::size_of::<AuthorFeatures>(), 16);
        assert_eq!(std::mem::size_of::<Option<AuthorFeatures>>(), 16);
    }

    #[test]
    fn label_set_membership() {
        let mut set = AuthorLabelSet::default();
        assert!(!set.has_label(AuthorLabel::NsfwHighRecall));
        set.insert(AuthorLabel::NsfwHighRecall);
        set.insert(AuthorLabel::DoNotAmplify);
        assert!(set.has_label(AuthorLabel::NsfwHighRecall));
        assert!(set.has_label(AuthorLabel::DoNotAmplify));
        assert!(!set.has_label(AuthorLabel::Compromised));
    }
}
