use crate::models::VfAction;
use crate::rules::{Rule, RuleContext};
use xai_visibility_filtering::models::FilteredReason;

pub struct ViewerBlocksAuthorRule;

impl Rule for ViewerBlocksAuthorRule {
    fn name(&self) -> &'static str {
        "ViewerBlocksAuthorRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        let candidate = context.candidate();
        if viewer.viewer_id().is_none() {
            return VfAction::Allow;
        }
        if candidate.relationship.viewer_blocks_author {
            return VfAction::Drop(FilteredReason::AuthorBlockViewer);
        }
        VfAction::Allow
    }
}

pub struct MutedRetweetsRule;

impl Rule for MutedRetweetsRule {
    fn name(&self) -> &'static str {
        "MutedRetweetsRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        let candidate = context.candidate();
        if viewer.viewer_id().is_none() {
            return VfAction::Allow;
        }
        if candidate.is_retweet() && candidate.relationship.viewer_mutes_retweets_from_author {
            return VfAction::Drop(FilteredReason::UnspecifiedReason);
        }
        VfAction::Allow
    }
}

pub struct ViewerMutesAuthorRule;

impl Rule for ViewerMutesAuthorRule {
    fn name(&self) -> &'static str {
        "ViewerMutesAuthorRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        let candidate = context.candidate();
        if viewer.viewer_id().is_none() {
            return VfAction::Allow;
        }
        if candidate.relationship.viewer_mutes_author {
            return VfAction::Drop(FilteredReason::ViewerMutesAuthor);
        }
        VfAction::Allow
    }
}

pub struct DropExclusiveTweetContentRule;

impl Rule for DropExclusiveTweetContentRule {
    fn name(&self) -> &'static str {
        "DropExclusiveTweetContentRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let viewer = context.viewer();
        let candidate = context.candidate();
        let Some(exc) = &candidate.exclusive_content else {
            return VfAction::Allow;
        };

        let Some(viewer_id) = viewer.viewer_id() else {
            return VfAction::Drop(FilteredReason::ExclusiveTweet);
        };

        if viewer_id == exc.conversation_author_id {
            return VfAction::Allow;
        }

        if exc.viewer_super_follows_author {
            return VfAction::Allow;
        }

        if !candidate.is_retweet() && context.is_author_viewer() {
            return VfAction::Allow;
        }

        VfAction::Drop(FilteredReason::ExclusiveTweet)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        ExclusiveContentFeatures, HydratedTweetCandidate, TweetFeatures, Viewer,
        ViewerAuthorRelationship, ViewerFeatures,
    };

    fn viewer(id: u64) -> ViewerFeatures {
        ViewerFeatures {
            viewer: Viewer::LoggedIn(id),
            ..Default::default()
        }
    }

    fn logged_out_viewer() -> ViewerFeatures {
        ViewerFeatures {
            viewer: Viewer::LoggedOut,
            ..Default::default()
        }
    }

    fn exclusive_candidate(
        tweet_id: u64,
        author_id: u64,
        root_author_id: u64,
    ) -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id,
            author_id,
            exclusive_content: Some(ExclusiveContentFeatures {
                conversation_author_id: root_author_id,
                viewer_super_follows_author: false,
            }),
            tweet_features: TweetFeatures::default(),
            ..Default::default()
        }
    }

    fn non_exclusive_candidate() -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            tweet_features: TweetFeatures::default(),
            ..Default::default()
        }
    }

    fn candidate_with_relationship(
        relationship: ViewerAuthorRelationship,
    ) -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            relationship,
            ..Default::default()
        }
    }

    #[test]
    fn viewer_blocks_author_drops() {
        let c = candidate_with_relationship(ViewerAuthorRelationship {
            viewer_blocks_author: true,
            ..Default::default()
        });
        assert!(matches!(
            ViewerBlocksAuthorRule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(FilteredReason::AuthorBlockViewer)
        ));
    }

    #[test]
    fn viewer_does_not_block_author_allows() {
        let c = candidate_with_relationship(ViewerAuthorRelationship::default());
        assert!(matches!(
            ViewerBlocksAuthorRule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn blocks_rule_allows_logged_out_viewer() {
        let c = candidate_with_relationship(ViewerAuthorRelationship {
            viewer_blocks_author: true,
            ..Default::default()
        });
        assert!(matches!(
            ViewerBlocksAuthorRule.evaluate(&crate::rules::test_context(&logged_out_viewer(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn viewer_mutes_author_drops() {
        let c = candidate_with_relationship(ViewerAuthorRelationship {
            viewer_mutes_author: true,
            ..Default::default()
        });
        assert!(matches!(
            ViewerMutesAuthorRule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(FilteredReason::ViewerMutesAuthor)
        ));
    }

    #[test]
    fn viewer_does_not_mute_author_allows() {
        let c = candidate_with_relationship(ViewerAuthorRelationship::default());
        assert!(matches!(
            ViewerMutesAuthorRule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn mutes_rule_allows_logged_out_viewer() {
        let c = candidate_with_relationship(ViewerAuthorRelationship {
            viewer_mutes_author: true,
            ..Default::default()
        });
        assert!(matches!(
            ViewerMutesAuthorRule.evaluate(&crate::rules::test_context(&logged_out_viewer(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn muted_retweets_drops_retweet_from_muting_viewer() {
        let mut c = candidate_with_relationship(ViewerAuthorRelationship {
            viewer_mutes_retweets_from_author: true,
            ..Default::default()
        });
        c.tweet_features.core.source_tweet_id = Some(99);
        assert!(matches!(
            MutedRetweetsRule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Drop(FilteredReason::UnspecifiedReason)
        ));
    }

    #[test]
    fn muted_retweets_allows_non_retweet() {
        let c = candidate_with_relationship(ViewerAuthorRelationship {
            viewer_mutes_retweets_from_author: true,
            ..Default::default()
        });
        assert!(matches!(
            MutedRetweetsRule.evaluate(&crate::rules::test_context(&viewer(999), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn muted_retweets_allows_logged_out_viewer() {
        let mut c = candidate_with_relationship(ViewerAuthorRelationship {
            viewer_mutes_retweets_from_author: true,
            ..Default::default()
        });
        c.tweet_features.core.source_tweet_id = Some(99);
        assert!(matches!(
            MutedRetweetsRule.evaluate(&crate::rules::test_context(&logged_out_viewer(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn non_exclusive_tweet_is_allowed() {
        let rule = DropExclusiveTweetContentRule;
        let action = rule.evaluate(&crate::rules::test_context(
            &viewer(999),
            &non_exclusive_candidate(),
        ));
        assert!(matches!(action, VfAction::Allow));
    }

    #[test]
    fn logged_out_viewer_drops_exclusive() {
        let rule = DropExclusiveTweetContentRule;
        let candidate = exclusive_candidate(1, 100, 100);
        let action = rule.evaluate(&crate::rules::test_context(
            &logged_out_viewer(),
            &candidate,
        ));
        assert!(matches!(
            action,
            VfAction::Drop(FilteredReason::ExclusiveTweet)
        ));
    }

    #[test]
    fn root_author_can_see_own_exclusive() {
        let rule = DropExclusiveTweetContentRule;
        let candidate = exclusive_candidate(1, 100, 100);
        let action = rule.evaluate(&crate::rules::test_context(&viewer(100), &candidate));
        assert!(matches!(action, VfAction::Allow));
    }

    #[test]
    fn super_follower_can_see_exclusive() {
        let rule = DropExclusiveTweetContentRule;
        let mut candidate = exclusive_candidate(1, 100, 100);
        candidate
            .exclusive_content
            .as_mut()
            .unwrap()
            .viewer_super_follows_author = true;
        let action = rule.evaluate(&crate::rules::test_context(&viewer(200), &candidate));
        assert!(matches!(action, VfAction::Allow));
    }

    #[test]
    fn non_super_follower_drops_exclusive() {
        let rule = DropExclusiveTweetContentRule;
        let candidate = exclusive_candidate(1, 100, 100);
        let action = rule.evaluate(&crate::rules::test_context(&viewer(200), &candidate));
        assert!(matches!(
            action,
            VfAction::Drop(FilteredReason::ExclusiveTweet)
        ));
    }

    #[test]
    fn reply_author_can_see_own_reply_in_exclusive_convo() {
        let rule = DropExclusiveTweetContentRule;
        let candidate = exclusive_candidate(2, 200, 100);
        let action = rule.evaluate(&crate::rules::test_context(&viewer(200), &candidate));
        assert!(matches!(action, VfAction::Allow));
    }

    #[test]
    fn retweet_author_cannot_self_view_exclusive() {
        let rule = DropExclusiveTweetContentRule;
        let mut candidate = exclusive_candidate(2, 200, 100);
        candidate.tweet_features.core.source_tweet_id = Some(99);
        let action = rule.evaluate(&crate::rules::test_context(&viewer(200), &candidate));
        assert!(matches!(
            action,
            VfAction::Drop(FilteredReason::ExclusiveTweet)
        ));
    }
}
