use crate::models::{TweetFeatures, VfAction};
use crate::rules::{Rule, RuleContext};
use xai_visibility_filtering::models::FilteredReason;

#[derive(Clone)]
pub struct TweetFlagDropRule {
    name: &'static str,
    flag: fn(&TweetFeatures) -> bool,
    reason: FilteredReason,
}

impl TweetFlagDropRule {
    pub const fn new(
        name: &'static str,
        flag: fn(&TweetFeatures) -> bool,
        reason: FilteredReason,
    ) -> Self {
        Self { name, flag, reason }
    }
}

impl Rule for TweetFlagDropRule {
    fn name(&self) -> &'static str {
        self.name
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        if (self.flag)(&context.candidate().tweet_features) {
            return VfAction::Drop(self.reason.clone());
        }
        VfAction::Allow
    }
}

pub const TWEET_NSFW_USER_DROP: TweetFlagDropRule = TweetFlagDropRule::new(
    "TweetNsfwUserDropRule",
    |t| t.nsfw.user,
    FilteredReason::ContainNsfwMedia,
);
pub const TWEET_NSFW_ADMIN_DROP: TweetFlagDropRule = TweetFlagDropRule::new(
    "TweetNsfwAdminDropRule",
    |t| t.nsfw.admin,
    FilteredReason::ContainNsfwMedia,
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        HydratedTweetCandidate, NsfwFeature, TweetFeatures, Viewer, ViewerFeatures,
    };

    fn candidate_with_nsfw_flags(user: bool, admin: bool) -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            tweet_features: TweetFeatures {
                nsfw: NsfwFeature { user, admin },
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn viewer(id: u64) -> ViewerFeatures {
        ViewerFeatures {
            viewer: Viewer::LoggedIn(id),
            ..Default::default()
        }
    }

    #[test]
    fn tweet_nsfw_user_drops() {
        let c = candidate_with_nsfw_flags(true, false);
        assert!(matches!(
            TWEET_NSFW_USER_DROP.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(FilteredReason::ContainNsfwMedia)
        ));
    }

    #[test]
    fn tweet_nsfw_user_unset_allows() {
        let c = candidate_with_nsfw_flags(false, false);
        assert!(matches!(
            TWEET_NSFW_USER_DROP.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn tweet_nsfw_user_drops_even_self_view() {
        let c = candidate_with_nsfw_flags(true, false);
        assert!(matches!(
            TWEET_NSFW_USER_DROP.evaluate(&crate::rules::test_context(&viewer(100), &c)),
            VfAction::Drop(FilteredReason::ContainNsfwMedia)
        ));
    }

    #[test]
    fn tweet_nsfw_admin_drops() {
        let c = candidate_with_nsfw_flags(false, true);
        assert!(matches!(
            TWEET_NSFW_ADMIN_DROP.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(FilteredReason::ContainNsfwMedia)
        ));
    }

    #[test]
    fn tweet_nsfw_admin_unset_allows() {
        let c = candidate_with_nsfw_flags(false, false);
        assert!(matches!(
            TWEET_NSFW_ADMIN_DROP.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn tweet_nsfw_admin_drops_even_self_view() {
        let c = candidate_with_nsfw_flags(false, true);
        assert!(matches!(
            TWEET_NSFW_ADMIN_DROP.evaluate(&crate::rules::test_context(&viewer(100), &c)),
            VfAction::Drop(FilteredReason::ContainNsfwMedia)
        ));
    }
}
