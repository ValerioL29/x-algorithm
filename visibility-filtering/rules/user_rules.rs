use crate::models::{AuthorFeatures, VfAction};
use crate::rules::{Rule, RuleContext};
use xai_visibility_filtering::models::FilteredReason;

#[derive(Clone)]
pub struct AuthorFlagDropRule {
    name: &'static str,
    flag: fn(&AuthorFeatures) -> bool,
    reason: FilteredReason,
}

impl AuthorFlagDropRule {
    pub const fn new(
        name: &'static str,
        flag: fn(&AuthorFeatures) -> bool,
        reason: FilteredReason,
    ) -> Self {
        Self { name, flag, reason }
    }
}

impl Rule for AuthorFlagDropRule {
    fn name(&self) -> &'static str {
        self.name
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let candidate = context.candidate();
        if (self.flag)(&candidate.author_features) && !context.is_author_viewer() {
            return VfAction::Drop(self.reason.clone());
        }
        VfAction::Allow
    }
}

pub const SUSPENDED_AUTHOR_DROP: AuthorFlagDropRule = AuthorFlagDropRule::new(
    "SuspendedAuthorRule",
    |a| a.is_suspended,
    FilteredReason::AuthorIsSuspended,
);
pub const DEACTIVATED_AUTHOR_DROP: AuthorFlagDropRule = AuthorFlagDropRule::new(
    "DeactivatedAuthorRule",
    |a| a.is_deactivated,
    FilteredReason::AuthorIsDeactivated,
);
pub const ERASED_AUTHOR_DROP: AuthorFlagDropRule = AuthorFlagDropRule::new(
    "ErasedAuthorRule",
    |a| a.is_erased,
    FilteredReason::AuthorAccountIsInactive,
);
pub const OFFBOARDED_AUTHOR_DROP: AuthorFlagDropRule = AuthorFlagDropRule::new(
    "OffboardedAuthorRule",
    |a| a.is_offboarded,
    FilteredReason::AuthorAccountIsInactive,
);
pub const NSFW_USER_AUTHOR_DROP: AuthorFlagDropRule = AuthorFlagDropRule::new(
    "DropNsfwUserAuthorRule",
    |a| a.is_nsfw_user,
    FilteredReason::ContainNsfwMedia,
);
pub const NSFW_ADMIN_AUTHOR_DROP: AuthorFlagDropRule = AuthorFlagDropRule::new(
    "DropNsfwAdminAuthorRule",
    |a| a.is_nsfw_admin,
    FilteredReason::ContainNsfwMedia,
);

pub struct ProtectedAuthorDropRule;

impl Rule for ProtectedAuthorDropRule {
    fn name(&self) -> &'static str {
        "ProtectedAuthorDropRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        let candidate = context.candidate();
        if candidate.author_features.is_protected
            && !context.is_author_viewer()
            && (viewer.viewer_is_logged_out() || !candidate.viewer_follows_author())
        {
            return VfAction::Drop(FilteredReason::AuthorIsProtected);
        }
        VfAction::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        AuthorFeatures, HydratedTweetCandidate, Viewer, ViewerAuthorRelationship, ViewerFeatures,
    };

    fn candidate_with_author(
        suspended: bool,
        deactivated: bool,
        protected: bool,
    ) -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            author_features: AuthorFeatures {
                is_suspended: suspended,
                is_deactivated: deactivated,
                is_protected: protected,
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
    fn suspended_author_drops() {
        let rule = SUSPENDED_AUTHOR_DROP;
        let c = candidate_with_author(true, false, false);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn suspended_author_self_view_allows() {
        let rule = SUSPENDED_AUTHOR_DROP;
        let c = candidate_with_author(true, false, false);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(100), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn deactivated_author_drops() {
        let rule = DEACTIVATED_AUTHOR_DROP;
        let c = candidate_with_author(false, true, false);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn protected_author_drops_non_follower() {
        let rule = ProtectedAuthorDropRule;
        let c = candidate_with_author(false, false, true);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn protected_author_allows_follower() {
        let rule = ProtectedAuthorDropRule;
        let mut c = candidate_with_author(false, false, true);
        c.relationship = ViewerAuthorRelationship {
            viewer_follows_author: true,
            ..Default::default()
        };
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn protected_author_drops_logged_out_viewer() {
        let rule = ProtectedAuthorDropRule;
        let c = candidate_with_author(false, false, true);
        let logged_out = ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..Default::default()
        };
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&logged_out, &c)),
            VfAction::Drop(FilteredReason::AuthorIsProtected)
        ));
    }

    #[test]
    fn protected_author_allows_self_view() {
        let rule = ProtectedAuthorDropRule;
        let c = candidate_with_author(false, false, true);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(100), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn erased_and_offboarded_drops_wire_flag_name_and_reason() {
        let cases: [(&AuthorFlagDropRule, &str, AuthorFeatures); 2] = [
            (
                &ERASED_AUTHOR_DROP,
                "ErasedAuthorRule",
                AuthorFeatures {
                    is_erased: true,
                    ..Default::default()
                },
            ),
            (
                &OFFBOARDED_AUTHOR_DROP,
                "OffboardedAuthorRule",
                AuthorFeatures {
                    is_offboarded: true,
                    ..Default::default()
                },
            ),
        ];
        for (rule, name, author_features) in cases {
            assert_eq!(rule.name(), name);
            let flagged = HydratedTweetCandidate {
                tweet_id: 1,
                author_id: 100,
                author_features,
                ..Default::default()
            };
            assert!(matches!(
                rule.evaluate(&crate::rules::test_context(&viewer(999), &flagged)),
                VfAction::Drop(FilteredReason::AuthorAccountIsInactive)
            ));
            let unflagged = HydratedTweetCandidate {
                tweet_id: 1,
                author_id: 100,
                ..Default::default()
            };
            assert!(matches!(
                rule.evaluate(&crate::rules::test_context(&viewer(999), &unflagged)),
                VfAction::Allow
            ));
        }
    }

    #[test]
    fn normal_author_allows() {
        let rule = SUSPENDED_AUTHOR_DROP;
        let c = candidate_with_author(false, false, false);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }

    fn candidate_with_nsfw_author(
        is_nsfw_user: bool,
        is_nsfw_admin: bool,
    ) -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            author_features: AuthorFeatures {
                is_nsfw_user,
                is_nsfw_admin,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn nsfw_user_author_drops() {
        let rule = NSFW_USER_AUTHOR_DROP;
        let c = candidate_with_nsfw_author(true, false);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn nsfw_user_author_self_view_allows() {
        let rule = NSFW_USER_AUTHOR_DROP;
        let c = candidate_with_nsfw_author(true, false);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(100), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn nsfw_admin_author_drops() {
        let rule = NSFW_ADMIN_AUTHOR_DROP;
        let c = candidate_with_nsfw_author(false, true);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn nsfw_admin_author_self_view_allows() {
        let rule = NSFW_ADMIN_AUTHOR_DROP;
        let c = candidate_with_nsfw_author(false, true);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(100), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn non_nsfw_author_allows() {
        let rule = NSFW_USER_AUTHOR_DROP;
        let c = candidate_with_nsfw_author(false, false);
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }
}
