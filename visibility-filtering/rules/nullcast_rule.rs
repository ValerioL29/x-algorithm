use crate::models::VfAction;
use crate::rules::{Rule, RuleContext};
use xai_visibility_filtering::models::FilteredReason;

pub struct NullcastedTweetDropRule;

impl Rule for NullcastedTweetDropRule {
    fn name(&self) -> &'static str {
        "NullcastedTweetDropRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let candidate = context.candidate();
        if candidate.is_nullcast() && !candidate.is_retweet() && !candidate.is_community_tweet() {
            return VfAction::Drop(FilteredReason::TweetIsNullcast);
        }
        VfAction::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        CoreFeature, HydratedTweetCandidate, TweetFeatures, Viewer, ViewerFeatures,
    };

    fn viewer() -> ViewerFeatures {
        ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            ..Default::default()
        }
    }

    #[test]
    fn nullcast_non_retweet_drops() {
        let rule = NullcastedTweetDropRule;
        let c = HydratedTweetCandidate {
            tweet_id: 1,
            tweet_features: TweetFeatures {
                is_nullcast: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(), &c)),
            VfAction::Drop(_)
        ));
    }

    #[test]
    fn nullcast_community_tweet_allows() {
        let rule = NullcastedTweetDropRule;
        let c = HydratedTweetCandidate {
            tweet_id: 1,
            tweet_features: TweetFeatures {
                is_nullcast: true,
                is_community_tweet: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn nullcast_retweet_allows() {
        let rule = NullcastedTweetDropRule;
        let c = HydratedTweetCandidate {
            tweet_id: 1,
            tweet_features: TweetFeatures {
                is_nullcast: true,
                core: CoreFeature {
                    source_tweet_id: Some(99),
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn non_nullcast_allows() {
        let rule = NullcastedTweetDropRule;
        let c = HydratedTweetCandidate {
            tweet_id: 1,
            tweet_features: TweetFeatures {
                is_nullcast: false,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(matches!(
            rule.evaluate(&crate::rules::test_context(&viewer(), &c)),
            VfAction::Allow
        ));
    }
}
