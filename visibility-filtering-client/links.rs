macro_rules! rules {
    ($path:literal) => {
        concat!("https://help.x.com/rules-and-policies/", $path)
    };
}
macro_rules! using {
    ($path:literal) => {
        concat!("https://help.x.com/using-x/", $path)
    };
}

pub const NOTICES_ON_TWITTER: &str = rules!("notices-on-x");
pub const ENFORCEMENT_OPTIONS: &str = rules!("enforcement-options");
pub const PUBLIC_INTEREST: &str = rules!("public-interest");
pub const TWITTER_RULES: &str = rules!("x-rules");
pub const ELECTION_INTEGRITY: &str = rules!("election-integrity-policy");
pub const CRISIS_MISINFORMATION: &str = rules!("crisis-misinformation");
pub const MEDICAL_MISINFORMATION: &str = rules!("medical-misinformation-policy");
pub const MANIPULATED_MEDIA: &str = rules!("manipulated-media");
pub const HATEFUL_CONDUCT_POLICY: &str = rules!("hateful-conduct-policy");
pub const ABUSIVE_BEHAVIOR: &str = rules!("abusive-behavior");
pub const ABUSIVE_PROFILE: &str = rules!("abusive-profile");
pub const ADULT_CONTENT: &str = rules!("adult-content");
pub const AGE_ASSURANCE: &str = rules!("age-assurance");
pub const LEGAL_DEMANDS_LOCAL_LAWS_WITHHELD: &str = rules!("post-withheld-by-country");

pub const COMMUNITIES: &str = using!("communities");
pub const SUPER_FOLLOWS: &str = using!("super-follows");
pub const PREMIUM_CONTENT: &str = "https://help.x.com/en/using-x/x-premium";
pub const DMCA_WITHHELD: &str = "https://help.x.com/articles/15795";

pub const PROFILE_SETTINGS: &str = "https://x.com/settings/profile";
pub const X_DOT_COM: &str = "https://x.com";

pub const BLURRED_MEDIA_TOMBSTONE_IMAGE: &str =
    "https://pbs.twimg.com/media/GxJIrSUagAAK-ZP?format=jpg&name=240x240";
