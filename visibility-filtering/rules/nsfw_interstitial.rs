use crate::models::{SafetyLabelType, VfAction};
use crate::rules::{Rule, RuleContext};
use xai_visibility_filtering::models::FilteredReason;

#[derive(Clone, Copy)]
pub struct NsfwMediaInterstitialRule {
    label: SafetyLabelType,
    name: &'static str,
}

impl NsfwMediaInterstitialRule {
    pub const fn new(label: SafetyLabelType, name: &'static str) -> Self {
        Self { label, name }
    }
}

impl Rule for NsfwMediaInterstitialRule {
    fn name(&self) -> &'static str {
        self.name
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        if context.has_tweet_safety_label(self.label)
            && !context.is_author_viewer()
            && !context.viewer_allows_sensitive_media()
        {
            return VfAction::Interstitial(FilteredReason::ContainNsfwMedia);
        }
        VfAction::Allow
    }
}

pub static NSFW_HIGH_PRECISION_INTERSTITIAL: NsfwMediaInterstitialRule =
    NsfwMediaInterstitialRule::new(
        SafetyLabelType::NSFW_HIGH_PRECISION,
        "NsfwHighPrecisionInterstitialRule",
    );

pub static GORE_AND_VIOLENCE_INTERSTITIAL: NsfwMediaInterstitialRule =
    NsfwMediaInterstitialRule::new(
        SafetyLabelType::GORE_AND_VIOLENCE_HIGH_PRECISION,
        "GoreAndViolenceInterstitialRule",
    );

pub static NSFW_CARD_IMAGE_INTERSTITIAL: NsfwMediaInterstitialRule = NsfwMediaInterstitialRule::new(
    SafetyLabelType::NSFW_CARD_IMAGE,
    "NsfwCardImageInterstitialRule",
);

pub struct NsfwAuthorInterstitialRule;

impl Rule for NsfwAuthorInterstitialRule {
    fn name(&self) -> &'static str {
        "NsfwAuthorInterstitialRule"
    }

    fn evaluate(&self, context: &RuleContext<'_>) -> VfAction {
        let candidate = context.candidate();
        if candidate.is_nsfw_flagged()
            && candidate.has_media()
            && !context.is_author_viewer()
            && !context.viewer_allows_sensitive_media()
        {
            return VfAction::Interstitial(FilteredReason::ContainNsfwMedia);
        }
        VfAction::Allow
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{
        AuthorFeatures, HydratedTweetCandidate, MediaFeature, NsfwFeature, SafetyLabel,
        SafetyLabelMap, TweetFeatures, Viewer, ViewerFeatures,
    };
    use std::collections::HashMap;

    fn viewer_default() -> ViewerFeatures {
        ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            allows_sensitive_media: false,
            ..Default::default()
        }
    }

    fn viewer_sensitive_opt_in() -> ViewerFeatures {
        ViewerFeatures {
            viewer: Viewer::LoggedIn(999),
            allows_sensitive_media: true,
            ..Default::default()
        }
    }

    fn candidate_with_label(label: SafetyLabelType) -> HydratedTweetCandidate {
        let mut labels = HashMap::new();
        labels.insert(label, SafetyLabel::default());
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            safety_labels: SafetyLabelMap::new(labels),
            ..Default::default()
        }
    }

    #[test]
    fn interstitial_blurs_non_opt_in() {
        let c = candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            NSFW_HIGH_PRECISION_INTERSTITIAL
                .evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Interstitial(_)
        ));
    }

    #[test]
    fn interstitial_allows_opt_in() {
        let c = candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        assert!(matches!(
            NSFW_HIGH_PRECISION_INTERSTITIAL
                .evaluate(&crate::rules::test_context(&viewer_sensitive_opt_in(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn interstitial_allows_self_view() {
        let mut c = candidate_with_label(SafetyLabelType::NSFW_HIGH_PRECISION);
        c.author_id = 999;
        assert!(matches!(
            NSFW_HIGH_PRECISION_INTERSTITIAL
                .evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn interstitial_allows_no_label() {
        let c = HydratedTweetCandidate {
            tweet_id: 1,
            ..Default::default()
        };
        assert!(matches!(
            NSFW_HIGH_PRECISION_INTERSTITIAL
                .evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Allow
        ));
    }

    fn nsfw_author_candidate() -> HydratedTweetCandidate {
        HydratedTweetCandidate {
            tweet_id: 1,
            author_id: 100,
            tweet_features: TweetFeatures {
                media: MediaFeature {
                    has_media: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            author_features: AuthorFeatures {
                is_nsfw_user: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn author_interstitial_blurs_non_opt_in() {
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(
                &viewer_default(),
                &nsfw_author_candidate()
            )),
            VfAction::Interstitial(_)
        ));
    }

    #[test]
    fn author_interstitial_allows_opt_in() {
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(
                &viewer_sensitive_opt_in(),
                &nsfw_author_candidate()
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn author_interstitial_allows_when_no_media() {
        let mut c = nsfw_author_candidate();
        c.tweet_features.media.has_media = false;
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Allow
        ));
    }

    fn nsfw_tweet_flag_candidate() -> HydratedTweetCandidate {
        let mut c = nsfw_author_candidate();
        c.author_features = AuthorFeatures::default();
        c.tweet_features.nsfw = NsfwFeature {
            user: true,
            admin: false,
        };
        c
    }

    #[test]
    fn tweet_flag_interstitial_blurs_non_opt_in() {
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(
                &viewer_default(),
                &nsfw_tweet_flag_candidate()
            )),
            VfAction::Interstitial(_)
        ));
    }

    #[test]
    fn tweet_admin_flag_interstitial_blurs_non_opt_in() {
        let mut c = nsfw_tweet_flag_candidate();
        c.tweet_features.nsfw = NsfwFeature {
            user: false,
            admin: true,
        };
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Interstitial(_)
        ));
    }

    #[test]
    fn both_flag_sources_interstitial_blurs_non_opt_in() {
        let mut c = nsfw_tweet_flag_candidate();
        c.author_features.is_nsfw_admin = true;
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Interstitial(_)
        ));
    }

    #[test]
    fn no_flags_allows() {
        let mut c = nsfw_tweet_flag_candidate();
        c.tweet_features.nsfw = NsfwFeature::default();
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Allow
        ));
    }

    #[test]
    fn tweet_flag_interstitial_allows_opt_in() {
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(
                &viewer_sensitive_opt_in(),
                &nsfw_tweet_flag_candidate()
            )),
            VfAction::Allow
        ));
    }

    #[test]
    fn tweet_flag_interstitial_allows_self_view() {
        let mut c = nsfw_tweet_flag_candidate();
        c.author_id = 999;
        assert!(matches!(
            NsfwAuthorInterstitialRule.evaluate(&crate::rules::test_context(&viewer_default(), &c)),
            VfAction::Allow
        ));
    }
}
