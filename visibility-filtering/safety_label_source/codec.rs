use xai_safety_label_store::error::SafetyLabelStoreError;
use xai_safety_label_store::types::{decode_lkey, SafetyLabelMap};
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::tweet_safety_label::SafetyLabel;

use super::proto;

use std::collections::HashMap;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};

use tracing::debug;

const VERSION: u8 = 0x01;

#[derive(Debug)]
pub enum CodecError {
    TooManyLabels { count: usize },
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyLabels { count } => {
                write!(f, "too many labels to encode: {count} (max {})", u16::MAX)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LkeyBytes(pub [u8; 4]);

#[derive(Debug, Clone)]
pub struct MvalBytes(pub Vec<u8>);

#[derive(Debug, Clone)]
pub struct RawSafetyLabel {
    pub lkey: LkeyBytes,
    pub mval: MvalBytes,
}

pub struct EncodedSafetyLabelMap(Vec<u8>);

impl EncodedSafetyLabelMap {
    pub fn into_bytes(self) -> Vec<u8> {
        self.0
    }
}

pub fn encode(items: &[RawSafetyLabel]) -> Result<EncodedSafetyLabelMap, CodecError> {
    let count: u16 = items
        .len()
        .try_into()
        .map_err(|_| CodecError::TooManyLabels { count: items.len() })?;
    let total: usize = 3 + items
        .iter()
        .map(|item| 8 + item.mval.0.len())
        .sum::<usize>();
    let mut buf = Vec::with_capacity(total);
    buf.push(VERSION);
    buf.extend_from_slice(&count.to_be_bytes());
    for item in items {
        buf.extend_from_slice(&item.lkey.0);
        buf.extend_from_slice(&(item.mval.0.len() as u32).to_be_bytes());
        buf.extend_from_slice(&item.mval.0);
    }
    Ok(EncodedSafetyLabelMap(buf))
}

pub fn decode(bytes: &[u8]) -> Option<SafetyLabelMap> {
    if bytes.len() < 3 {
        return None;
    }
    if bytes[0] != VERSION {
        return None;
    }
    let count = u16::from_be_bytes([bytes[1], bytes[2]]) as usize;
    let mut cursor = 3;
    let mut map = HashMap::with_capacity(count);
    for _ in 0..count {
        if cursor + 8 > bytes.len() {
            return None;
        }
        let lkey = &bytes[cursor..cursor + 4];
        cursor += 4;
        let len = u32::from_be_bytes(bytes[cursor..cursor + 4].try_into().ok()?) as usize;
        cursor += 4;
        if cursor + len > bytes.len() {
            return None;
        }
        let mval = &bytes[cursor..cursor + len];
        cursor += len;

        let label_type = match decode_lkey(lkey) {
            Ok(lt) => lt,
            Err(SafetyLabelStoreError::InvalidLabelType(_)) => continue,
            Err(_) => return None,
        };
        let label = match xai_x_thrift::deserialize_mval(mval) {
            Ok(l) => l,
            Err(e) => {
                debug!(label_type = ?label_type, error = %e, "skipping label value with corrupt mval from twemcache");
                map.insert(label_type, SafetyLabel::default());
                continue;
            }
        };
        map.insert(label_type, label);
    }
    Some(map)
}

pub fn decode_mval_payload(bytes: &[u8]) -> Option<vf_pb::SafetyLabelMap> {
    xai_safety_label_store::mval_safety_label_map::decode_mval(bytes)
        .map(|labels| proto::label_map_to_proto(&labels))
}

pub(crate) enum DecodeAttempt {
    Success(vf_pb::SafetyLabelMap),
    Failure(SafetyLabelStoreError),
    Panic,
}

pub(crate) fn decode_raw_labels(items: &[RawSafetyLabel]) -> DecodeAttempt {
    match catch_unwind(AssertUnwindSafe(|| decode_raw_labels_inner(items))) {
        Ok(Ok(map)) => DecodeAttempt::Success(proto::label_map_to_proto(&map)),
        Ok(Err(e)) => DecodeAttempt::Failure(e),
        Err(_) => DecodeAttempt::Panic,
    }
}

fn decode_raw_labels_inner(
    items: &[RawSafetyLabel],
) -> Result<SafetyLabelMap, SafetyLabelStoreError> {
    let mut map = HashMap::with_capacity(items.len());
    for item in items {
        let label_type = match decode_lkey(&item.lkey.0) {
            Ok(lt) => lt,
            Err(SafetyLabelStoreError::InvalidLabelType(_)) => continue,
            Err(e) => return Err(e),
        };
        let label = match xai_x_thrift::deserialize_mval(&item.mval.0) {
            Ok(l) => l,
            Err(e) => {
                debug!(label_type = ?label_type, error = %e, "skipping label value with corrupt mval from manhattan");
                SafetyLabel::default()
            }
        };
        map.insert(label_type, label);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use xai_safety_label_store::types::encode_lkey;
    use xai_x_thrift::tweet_safety_label::SafetyLabelType;

    fn minimal_mval() -> Vec<u8> {
        vec![0x0C, 0x00, 0x04, 0x00, 0x00]
    }

    fn mval_with_score() -> Vec<u8> {
        let mut buf = vec![0x0C, 0x00, 0x04];
        buf.extend_from_slice(&[0x04, 0x00, 0x01]);
        buf.extend_from_slice(&0.9f64.to_bits().to_be_bytes());
        buf.push(0x00);
        buf.push(0x00);
        buf
    }

    fn raw_label(lt: SafetyLabelType, mval: Vec<u8>) -> RawSafetyLabel {
        RawSafetyLabel {
            lkey: LkeyBytes(encode_lkey(lt)),
            mval: MvalBytes(mval),
        }
    }

    #[test]
    fn encode_decode_roundtrip_empty() {
        let blob = encode(&[]).unwrap();
        assert!(blob.0.len() == 3);
        let map = decode(&blob.0).unwrap();
        assert!(map.is_empty());
    }

    #[test]
    fn encode_decode_roundtrip_one_label() {
        let items = vec![raw_label(SafetyLabelType::SPAM, minimal_mval())];
        let blob = encode(&items).unwrap();
        let map = decode(&blob.0).unwrap();
        assert!(map.len() == 1);
        assert!(map.contains_key(&SafetyLabelType::SPAM));
    }

    #[test]
    fn encode_decode_roundtrip_multiple_labels() {
        let items = vec![
            raw_label(SafetyLabelType::SPAM, minimal_mval()),
            raw_label(SafetyLabelType::BOUNCE, mval_with_score()),
            raw_label(SafetyLabelType::NSFW_HIGH_PRECISION, minimal_mval()),
        ];
        let blob = encode(&items).unwrap();
        let map = decode(&blob.0).unwrap();
        assert!(map.len() == 3);
        assert!(map.contains_key(&SafetyLabelType::SPAM));
        assert!(map.contains_key(&SafetyLabelType::BOUNCE));
        assert!(map.contains_key(&SafetyLabelType::NSFW_HIGH_PRECISION));
        assert_eq!(
            map[&SafetyLabelType::BOUNCE].score.map(|f| f.into_inner()),
            Some(0.9)
        );
    }

    #[test]
    fn decode_returns_none_on_empty() {
        assert!(decode(&[]).is_none());
    }

    #[test]
    fn decode_returns_none_on_garbage() {
        assert!(decode(&[0xff; 16]).is_none());
    }

    #[test]
    fn decode_returns_none_on_truncated() {
        let items = vec![raw_label(SafetyLabelType::SPAM, minimal_mval())];
        let blob = encode(&items).unwrap();
        assert!(decode(&blob.0[..blob.0.len() - 2]).is_none());
    }

    #[test]
    fn decode_returns_none_on_wrong_version() {
        let mut blob = encode(&[]).unwrap().0;
        blob[0] = 0x02;
        assert!(decode(&blob).is_none());
    }

    #[test]
    fn decode_includes_unknown_label_types() {
        let unknown_lkey = (999i32 ^ i32::MIN).to_be_bytes();
        let items = vec![
            RawSafetyLabel {
                lkey: LkeyBytes(unknown_lkey),
                mval: MvalBytes(minimal_mval()),
            },
            raw_label(SafetyLabelType::SPAM, minimal_mval()),
        ];
        let blob = encode(&items).unwrap();
        let map = decode(&blob.0).unwrap();
        assert!(map.len() == 2);
        assert!(map.contains_key(&SafetyLabelType::SPAM));
        assert!(map.contains_key(&SafetyLabelType::from(999)));
    }

    #[track_caller]
    fn decoded(items: &[RawSafetyLabel]) -> vf_pb::SafetyLabelMap {
        match decode_raw_labels(items) {
            DecodeAttempt::Success(map) => map,
            DecodeAttempt::Failure(e) => panic!("expected Success, got Failure({e})"),
            DecodeAttempt::Panic => panic!("expected Success, got Panic"),
        }
    }

    #[test]
    fn decode_raw_labels_matches_decode() {
        let items = vec![
            raw_label(SafetyLabelType::SPAM, minimal_mval()),
            raw_label(SafetyLabelType::BOUNCE, mval_with_score()),
        ];
        let blob = encode(&items).unwrap();
        let from_decode = proto::label_map_to_proto(&decode(&blob.0).unwrap());
        let from_raw = decoded(&items);
        assert!(from_decode == from_raw);
    }

    #[test]
    fn decode_raw_labels_preserves_label_type_on_corrupt_mval() {
        let items = vec![
            raw_label(SafetyLabelType::SPAM, minimal_mval()),
            raw_label(SafetyLabelType::BOUNCE, vec![0xDE, 0xAD]),
            raw_label(SafetyLabelType::NSFW_HIGH_PRECISION, mval_with_score()),
        ];
        let map = decoded(&items);
        assert_eq!(map.labels.len(), 3);
        assert!(map.labels.contains_key(&i32::from(SafetyLabelType::SPAM)));
        assert!(map.labels.contains_key(&i32::from(SafetyLabelType::BOUNCE)));
        assert!(map
            .labels
            .contains_key(&i32::from(SafetyLabelType::NSFW_HIGH_PRECISION)));
        assert_eq!(
            map.labels[&i32::from(SafetyLabelType::BOUNCE)],
            vf_pb::SafetyLabel::default()
        );
    }

    #[test]
    fn decode_preserves_label_type_on_corrupt_mval() {
        let items = vec![
            raw_label(SafetyLabelType::SPAM, minimal_mval()),
            raw_label(SafetyLabelType::BOUNCE, vec![0xDE, 0xAD]),
            raw_label(SafetyLabelType::NSFW_HIGH_PRECISION, minimal_mval()),
        ];
        let blob = encode(&items).unwrap();
        let map = decode(&blob.0).unwrap();
        assert_eq!(map.len(), 3);
        assert!(map.contains_key(&SafetyLabelType::SPAM));
        assert!(map.contains_key(&SafetyLabelType::BOUNCE));
        assert!(map.contains_key(&SafetyLabelType::NSFW_HIGH_PRECISION));
        assert_eq!(map[&SafetyLabelType::BOUNCE], SafetyLabel::default());
    }
}
